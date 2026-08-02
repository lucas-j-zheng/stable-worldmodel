#!/bin/bash
#SBATCH -J swm-e6p5-path
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -x gpu3005
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# ============================================================================
# E6.5 follow-up — is the latent MEAN DRIFT real, or a measurement-path artifact?
#
# The moment-penalty arms (4367358) produced a contradiction that invalidates
# the premise of the whole cell: the penalty drove the TRAINING-batch mean to
# ~0.14 (fit/mean_loss 1e-4, val 2.1) while the offline cache of the SAME
# checkpoint still measured mean_norm 9.87 — and a per-sample encoder's batch
# mean is an unbiased estimate of its population mean, so both cannot describe
# the same (encoder, inputs, mode). Until this is resolved, the E6.0a mean-drift
# mechanism — and every conclusion resting on it — is unfounded.
#
# PRE-REGISTERED READS (probe: same frames, three encode paths)
#   transform effect >> mode effect, on the FINE-TUNED encoder only  =>  the
#     "drift" is a train/inference PREPROCESSING MISMATCH that fine-tuning made
#     the encoder sensitive to. That is a pipeline bug, not a property of e2e,
#     and it would also be a live candidate for the ~2x planning loss (the
#     planner encodes through a path the encoder was never trained on).
#   mode effect dominates  =>  train/eval module-mode asymmetry (dropout/
#     droppath) — same conclusion class, different fix.
#   frozen encoder shows the SAME path sensitivity  =>  it is a property of the
#     pipeline, not of fine-tuning; then the frozen "mean_norm ~0" reference is
#     itself path-specific and the drift comparison was never apples-to-apples.
#   all paths agree (<~1.0 apart) on both encoders  =>  the discrepancy is in
#     the loss bookkeeping instead; re-derive from fit/mean_loss directly.
#
# Also repairs two legs of 4367357:
#   (a) A1's frozen reference died opening pusht_latent.lance (TorchCodec vs
#       torch 2.10 — that cache carries a video column). Rebuild a video-free
#       frozen cache and measure it, so "frozen within_std ~0.97" is a FRESH
#       number rather than a banked one, since A1's revision of Q1 leans on it.
#   (b) the whitened-cost benign-intervention control then had no stats file
#       and crashed; rerun it now that frozen_base.npz exists.
# ============================================================================

set -uo pipefail
cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate
export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs "$STABLEWM_HOME/latstats"
STATS=$STABLEWM_HOME/latstats

# ------------------------------------------------- (a) frozen reference, rebuilt
echo "[e6p5b] ===== frozen base cache (video-free) + latstats ====="
python scripts/data/cache_latents.py \
  lewm_checkpoint=lewm_pusht/weights.pt \
  source=pusht_expert_train.h5 out_name=pusht_base_frozen.lance \
  device=cuda batch_size=256 || echo "[e6p5b] frozen cache FAILED"

python scripts/data/latent_marginal_stats.py \
  --dataset pusht_base_frozen.lance --n-episodes 50 \
  --dump-stats "$STATS/frozen_base.npz" || echo "[e6p5b] frozen latstats FAILED"

# ------------------------------------------------------------ (P) the path probe
echo "[e6p5b] ===== path attribution: frozen (control) vs fine-tuned ====="
for pair in "lewm_pusht/weights.pt:frozen" \
            "e2w_x0_base_ls0p03/weights_epoch_2.pt:warm_ls0p03" \
            "e6m_mean_ls0p03/weights_epoch_2.pt:moment_w1"; do
  ckpt=${pair%%:*}; tag=${pair##*:}
  python scripts/data/latent_path_probe.py \
    --encoder "$ckpt" --tag "$tag" --n-episodes 10 \
    || echo "[e6p5b] path probe $tag FAILED"
done

# --------------------------------------- (b) the whitened-cost control, repaired
echo "[e6p5b] ===== whitened frozen x0 control (must stay ~20-22) ====="
FROZEN_CKPT=$(cd "$STABLEWM_HOME/checkpoints" && ls -t dyn_x0_pusht_ps1/*.pt 2>/dev/null | head -1)
if [ -n "$FROZEN_CKPT" ] && [ -e "$STATS/frozen_base.npz" ]; then
  for es in 42 123; do
    echo "[e6p5b] ==== EVAL whiten_frozen_x0_ctrl @1 (n=50, eval seed $es) ===="
    tmp="logs/_e6p5b_frozenctrl_${es}.log"
    if python scripts/plan/eval_wm.py --config-name pusht_diffusion \
        policy="$FROZEN_CKPT" cache_dir="$LOCAL_DATASET_DIR" seed=$es \
        "+model_overrides={num_inference_steps:1,cost_whiten:true,cost_stats_path:'$STATS/frozen_base.npz'}" \
        > "$tmp" 2>&1; then
      grep -iE "success_rate" "$tmp" | tail -1
    else
      echo "[e6p5b] EVAL FAILED — tail:"; tail -15 "$tmp"
    fi
  done
else
  echo "[e6p5b] control SKIPPED (ckpt or stats missing)"
fi

echo "DONE-E6P5-PATH"
