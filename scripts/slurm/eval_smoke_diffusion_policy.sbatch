#!/bin/bash
#SBATCH -J ldm-dp-evalsmoke
#SBATCH -p gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 00:40:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate
export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

# Validate the closed-loop policy->solver->env integration end-to-end on 1-door
# TwoRoom (in-dist for lewm_tworoom) using the 3-epoch dp_smoke checkpoint. Not a
# science result (under-trained, unimodal greedy data) -- just confirms actions
# flow and episodes run without error.
echo "[eval-smoke] DiffusionPolicy closed-loop ..."
python scripts/plan/eval_wm.py --config-name diffusion_policy_eval \
  policy=dp_smoke/weights_epoch_3.pt cache_dir="$LOCAL_DATASET_DIR" \
  eval.num_eval=10
echo "[eval-smoke] result:"
cat "$STABLEWM_HOME/checkpoints/dp_smoke/diffusion_policy_eval_results.txt" 2>/dev/null | tail -4
