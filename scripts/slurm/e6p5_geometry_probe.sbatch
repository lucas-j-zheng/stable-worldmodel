#!/bin/bash
#SBATCH -J swm-e6p5-geom
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -x gpu3005
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# ============================================================================
# E6.5 stage A — LATENT-GEOMETRY probe + whitened-cost evals (NO retraining)
#
# Context (verdicts doc 2026-07-23, final section): both E6.0a warm-start arms
# came back with latent mean_norm ~10 (frozen ~0) while global_std stayed ~1,
# and planning sits at ~half the frozen x0 baseline (10/14 and 8/6 @1 vs 20-22).
# Mean drift is the standing candidate mechanism. This stage splits it into two
# measurable halves before spending GPU-hours on retraining:
#
#   A1  Is "global_std ~1" real anti-collapse, or is it propped up by the mean
#       offset?  global_std is taken over the FLATTENED array, so it mixes
#       within-dim spread with cross-dim mean spread. From the banked numbers
#       (||mu||^2 ~ 105 over 192 dims) roughly HALF the reported variance may be
#       static offset, i.e. true per-dim std ~0.6 vs frozen ~0.97 — a 30% shrink
#       that the Q1 PASS would have hidden. --dump-stats also writes the per-dim
#       stats that A3 needs.
#   A2  Did the drift distort what the planning cost SEES? D3b (2026-07-24)
#       refuted block-blindness for the FROZEN and e2c encoders, but was never
#       run on the warm-start encoders. Re-run it there.
#   A3  Is the residual planning deficit just cost GEOMETRY? The criterion is a
#       plain L2 to the goal latent — correct only while the marginal is
#       isotropic. Per-dim stds now span 0.20-1.76, so plain L2 up-weights
#       whichever dims stayed wide. Whiten the distance by the per-dim std
#       (inference-only, new cost_whiten override) and re-eval.
#
# PRE-REGISTERED READS
#   A1  mean_share_of_global_var >~ 40%  =>  Q1's "anti-collapse fully works"
#       is REVISED to "partial collapse masked by drift"; the anti-collapse
#       machinery is weaker than scored and SIGReg needs the location term
#       regardless of what A3 says.
#   A2  agent:block coefficient ratio flips vs frozen (0.124:0.493) => the
#       fine-tuned encoder lost block structure => the deficit is representation
#       content, not geometry, and stage B will not fix it.
#   A3  whitened @1 recovers toward 20 on either warm arm => the deficit is
#       COST GEOMETRY, fixable at inference with zero retraining (and stage B's
#       training-side penalty is then optional/secondary).
#       No movement (within +-4 of the plain numbers) => geometry is not it;
#       stage B's location/scale control is the live test.
#   Control: the same whitening on the FROZEN x0 baseline must be a no-op
#       (frozen latents are already isotropic). If it moves the frozen number,
#       the intervention itself is confounded and A3 is uninterpretable.
# ============================================================================

set -uo pipefail
cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate
export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs "$STABLEWM_HOME/latstats"

STATS=$STABLEWM_HOME/latstats

# ---------------------------------------------------------------- A1: geometry
echo "[e6p5] ===== A1: latent marginal decomposition (frozen vs warm arms) ====="
for pair in "pusht_latent.lance:frozen_base" \
            "pusht_base_e2w_x0_base_ls0p1.lance:warm_ls0p1" \
            "pusht_base_e2w_x0_base_ls0p03.lance:warm_ls0p03"; do
  ds=${pair%%:*}; tag=${pair##*:}
  python scripts/data/latent_marginal_stats.py \
    --dataset "$ds" --n-episodes 50 --dump-stats "$STATS/${tag}.npz" \
    || echo "[e6p5] A1 $tag FAILED"
done

# ------------------------------------------------------- A2: cost--goal content
echo "[e6p5] ===== A2: cost-goal alignment on the WARM-START encoders ====="
echo "[e6p5] reference (banked 2026-07-24): frozen_base agent 0.124 / block 0.493"
for ls in ls0p1 ls0p03; do
  python scripts/data/cost_goal_alignment.py \
    --dataset pusht_expert_train.h5 \
    --encoder "e2w_x0_base_${ls}/weights_epoch_2.pt" \
    --tag "warm_${ls}_base" --n-episodes 40 || echo "[e6p5] A2 $ls FAILED"
done

# --------------------------------------------------------- A3: whitened @1 eval
# @1 is the primary readout program-wide (universal sampling penalty, arcs 1-3).
run_eval () {  # $1=ckpt  $2=label  $3=stats npz  ("" = no whitening)
  local ckpt=$1 label=$2 npz=$3
  [ -e "$STABLEWM_HOME/checkpoints/$ckpt" ] || { echo "[e6p5] SKIP $label (no ckpt)"; return; }
  local ov="{num_inference_steps:1}"
  [ -n "$npz" ] && ov="{num_inference_steps:1,cost_whiten:true,cost_stats_path:'$npz'}"
  for es in 42 123; do
    echo "[e6p5] ==== EVAL $label @1 (n=50, eval seed $es) ===="
    local tmp="logs/_e6p5_${label}_${es}.log"
    if python scripts/plan/eval_wm.py --config-name pusht_diffusion \
        policy="$ckpt" cache_dir="$LOCAL_DATASET_DIR" seed=$es \
        "+model_overrides=$ov" > "$tmp" 2>&1; then
      grep -iE "success_rate" "$tmp" | tail -1
    else
      echo "[e6p5] EVAL FAILED — tail:"; tail -15 "$tmp"
    fi
  done
}

echo "[e6p5] ===== A3: whitened planning cost (baselines: warm ls0p03 10/14, ls0p1 8/6, frozen x0 20-22) ====="
run_eval "e2w_x0_base_ls0p03/weights_epoch_2.pt" "whiten_warm_ls0p03_ep2" "$STATS/warm_ls0p03.npz"
run_eval "e2w_x0_base_ls0p1/weights_epoch_2.pt"  "whiten_warm_ls0p1_ep2"  "$STATS/warm_ls0p1.npz"

# Benign-intervention control on the frozen baseline (must stay ~20-22).
FROZEN_CKPT=$(cd "$STABLEWM_HOME/checkpoints" && ls -t dyn_x0_pusht_ps1/*.pt 2>/dev/null | head -1)
if [ -n "$FROZEN_CKPT" ]; then
  run_eval "$FROZEN_CKPT" "whiten_frozen_x0_ctrl" "$STATS/frozen_base.npz"
else
  echo "[e6p5] no dyn_x0_pusht_ps1 checkpoint — control SKIPPED"
fi

echo "DONE-E6P5-GEOM"
