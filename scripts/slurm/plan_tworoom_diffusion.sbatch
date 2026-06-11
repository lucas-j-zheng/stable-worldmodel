#!/bin/bash
#SBATCH -J plan-tworoom-ldm
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate

export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
export DIFF_CKPT=${DIFF_CKPT:-latent_diffusion_tworoom_base/weights_epoch_25.pt}

mkdir -p logs

echo "[plan-tworoom-ldm] DIFF_CKPT=$DIFF_CKPT"

python scripts/plan/eval_wm.py \
  --config-name tworoom_diffusion \
  policy="$DIFF_CKPT" \
  cache_dir="$LOCAL_DATASET_DIR"

cat "$STABLEWM_HOME/checkpoints/$(dirname "$DIFF_CKPT")/tworoom_diffusion_results.txt"
