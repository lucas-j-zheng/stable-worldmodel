#!/bin/bash
#SBATCH -J ldm-cache
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate

export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
export TWOROOM_DATA=${TWOROOM_DATA:-tworoom_expert.lance}
export LEWM_CKPT=${LEWM_CKPT:-lewm_tworoom/weights_epoch_300.pt}
export TWOROOM_LATENT_DATA=${TWOROOM_LATENT_DATA:-tworoom_latent_e300.lance}

mkdir -p logs "$STABLEWM_HOME/datasets" "$STABLEWM_HOME/checkpoints"

echo "[ldm-cache] LEWM_CKPT=$LEWM_CKPT"
echo "[ldm-cache] TWOROOM_DATA=$TWOROOM_DATA -> $TWOROOM_LATENT_DATA"

python scripts/data/cache_latents.py \
  lewm_checkpoint="$LEWM_CKPT" \
  source="$TWOROOM_DATA" \
  out_name="$TWOROOM_LATENT_DATA" \
  device=cuda \
  batch_size=256
