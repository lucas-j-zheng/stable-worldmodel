#!/bin/bash
#SBATCH -J ldm-dp-reeval
#SBATCH -p gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 00:55:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate
export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

# Diagnostic (eval only, no retrain): the mm comparison gave 0%/0%. Re-eval the
# trained dp_mm / mlp_mm at goal offsets closer to the training window to test the
# goal-horizon-mismatch hypothesis. If DP rises above 0 and above MLP, that's the
# multimodality win; if both stay 0 at every offset, the 2-door eval/env setup is
# the problem (next: sanity-check a known-good policy on this env).
for GO in 8 12 25; do
  echo "==== goal_offset=$GO :: DiffusionPolicy ===="
  python scripts/plan/eval_wm.py --config-name diffusion_policy_eval_mm \
    policy=dp_mm/weights_epoch_250.pt cache_dir="$LOCAL_DATASET_DIR" \
    eval.goal_offset_steps=$GO eval.num_eval=20 2>&1 | grep -iE "success_rate" | tail -1
  echo "==== goal_offset=$GO :: MLP ===="
  python scripts/plan/eval_wm.py --config-name diffusion_policy_eval_mm \
    policy=mlp_mm/weights_epoch_250.pt cache_dir="$LOCAL_DATASET_DIR" \
    eval.goal_offset_steps=$GO eval.num_eval=20 2>&1 | grep -iE "success_rate" | tail -1
done
echo "DONE-REEVAL"
