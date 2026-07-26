#!/bin/bash
#SBATCH -J swm-eval-replay
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# EVAL REPLAY (2026-07-24). The upstream plan-refactor (#282) added
# cfg.objective to eval_wm.py; the custom *_diffusion eval configs lacked the
# default, so EVERY eval in the 07-23 cells crashed (training/latstats fine).
# Configs are fixed; this replays just the evals over the surviving ckpts:
#   - E6.1 pareto Q3: w in {0p0, 0p02, 0p045, 0p09} @1/@20, base-pusht
#   - E6.0a warmstart ls0p1: epochs 1,2 @1/@20, base-pusht  (decisive Q3 test)
#   - 5c ps3 refill: BASE + CSLIP @1/@20
set -uo pipefail
cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate
export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

run_eval () {  # $1=ckpt $2=label $3=steps $4=extra
  local ckpt=$1 label=$2 st=$3 extra=${4:-}
  [ -e "$STABLEWM_HOME/checkpoints/$ckpt" ] || { echo "[replay] SKIP $label (no $ckpt)"; return; }
  for es in 42 123; do
    echo "[replay] ==== EVAL $label @${st} steps (n=50, eval seed $es) ===="
    local tmp="logs/_replay_${label}_${st}_${es}.log"
    if python scripts/plan/eval_wm.py --config-name pusht_diffusion \
        policy=$ckpt cache_dir="$LOCAL_DATASET_DIR" seed=$es \
        "+model_overrides={num_inference_steps:$st}" $extra > "$tmp" 2>&1; then
      grep -iE "success_rate" "$tmp" | tail -1
    else
      echo "[replay] EVAL FAILED — tail:"; tail -12 "$tmp"
    fi
  done
}

CSLIP="+world.contact_slip_scale=0.5 eval.dataset_name=pusht_cslip05.lance"

# E6.1 pareto (base-pusht Q3; frozen refs v-diff 10 / x0 20-22 / MSE 30.7)
for W in w0p0 w0p02 w0p045 w0p09 w0p18; do
  CK="e2s_diff_cslip05_${W}_ps1/weights_epoch_5.pt"
  run_eval "$CK" "pareto_${W}" 1
  run_eval "$CK" "pareto_${W}" 20
done

# E6.0a warmstart (frozen x0 baseline 20-22 @1; parity vs floor is the verdict)
for LS in ls0p1 ls0p03; do
  for EPN in 1 2; do
    CK="e2w_x0_base_${LS}/weights_epoch_${EPN}.pt"
    run_eval "$CK" "warm_${LS}_ep${EPN}" 1
    run_eval "$CK" "warm_${LS}_ep${EPN}" 20
  done
done

# 5c ps3 refill (matched + base)
CK="e2c_diff_cslip05_ps3/weights_epoch_5.pt"
run_eval "$CK" "e2c_ps3_base"  1
run_eval "$CK" "e2c_ps3_base"  20
run_eval "$CK" "e2c_ps3_cslip" 1  "$CSLIP"
run_eval "$CK" "e2c_ps3_cslip" 20 "$CSLIP"

echo "DONE-EVAL-REPLAY"
