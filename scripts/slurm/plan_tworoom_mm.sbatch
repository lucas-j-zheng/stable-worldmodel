#!/bin/bash
#SBATCH -J ldm-plan-mm
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate
export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

DIFF_CKPT=${DIFF_CKPT:-latent_diffusion_tworoom_mm_v/weights_epoch_250.pt}
DET_CKPT=${DET_CKPT:-lewm_tworoom/weights_epoch_300.pt}

# Closed-loop on the 2-door multimodal TwoRoom: diffusion D-MPC vs the
# deterministic LeWM predictor (regular MPC). Same CEM planner, same env.
echo "[plan-mm] DIFFUSION: $DIFF_CKPT"
python scripts/plan/eval_wm.py --config-name tworoom_mm_diffusion \
  policy="$DIFF_CKPT" cache_dir="$LOCAL_DATASET_DIR"
echo "[plan-mm] DETERMINISTIC: $DET_CKPT"
python scripts/plan/eval_wm.py --config-name tworoom_mm_diffusion \
  policy="$DET_CKPT" cache_dir="$LOCAL_DATASET_DIR"

echo "==== DIFFUSION (mm) ===="
cat "$STABLEWM_HOME/checkpoints/$(dirname "$DIFF_CKPT")/tworoom_mm_diffusion_results.txt" | tail -5
echo "==== DETERMINISTIC ===="
cat "$STABLEWM_HOME/checkpoints/$(dirname "$DET_CKPT")/tworoom_mm_diffusion_results.txt" | tail -5
