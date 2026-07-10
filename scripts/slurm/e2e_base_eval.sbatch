#!/bin/bash
#SBATCH -J ldm-e2e-baseeval
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 05:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -uo pipefail
cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate
export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

# ARC 5 control-attribution: evaluate the 6 e2e checkpoints on BASE PushT
# (no contact slip, the exact ARC-2 eval setup) where frozen-encoder baselines
# exist: v-diff 10.0, x0 20-22, clean-MSE 30.7. Answers "did joint e2e training
# help or hurt PLANNING" independent of the (possibly floor-hard) slip env.
# NB: the e2e models were TRAINED on contact-slip data — a distribution-shift
# caveat vs ARC2 models trained on base-pusht data; note it in the writeup.
TAG=cslip05
EP=5

run_eval () {  # $1=name $2=steps(""=tmse deterministic)
  local name=$1 st=${2:-}
  local extra=""
  [ -n "$st" ] && extra="+model_overrides={num_inference_steps:$st}"
  for es in 42 123; do
    echo "[e2e-base] ==== EVAL $name steps=${st:-det} BASE-PUSHT (n=50, eval seed $es) ===="
    local tmp="logs/_bev_${name}_${st:-det}_${es}.log"
    if python scripts/plan/eval_wm.py --config-name pusht_diffusion \
        policy=${name}/weights_epoch_$EP.pt cache_dir="$LOCAL_DATASET_DIR" \
        seed=$es $extra > "$tmp" 2>&1; then
      grep -iE "success_rate" "$tmp" | tail -1
    else
      echo "[e2e-base] EVAL FAILED — traceback tail:"
      tail -25 "$tmp"
    fi
  done
}

for s in 1 2 3; do
  run_eval "e2e_diff_${TAG}_ps${s}" 1
  run_eval "e2e_diff_${TAG}_ps${s}" 20
  run_eval "e2e_tmse_${TAG}_ps${s}" ""
done
echo "DONE-E2E-BASE-EVAL"
