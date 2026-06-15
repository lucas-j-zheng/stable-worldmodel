#!/bin/bash
#SBATCH -J plan-pusht
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
# Diffusion checkpoint, or a LeWM repo/ckpt for the deterministic baseline.
export POLICY=${POLICY:-latent_diffusion_pusht_v/weights_epoch_5.pt}

mkdir -p logs

echo "[plan-pusht] POLICY=$POLICY"

python scripts/plan/eval_wm.py \
  --config-name pusht_diffusion \
  policy="$POLICY" \
  cache_dir="$LOCAL_DATASET_DIR"
