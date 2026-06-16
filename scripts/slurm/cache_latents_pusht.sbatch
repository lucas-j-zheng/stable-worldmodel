#!/bin/bash
#SBATCH -J ldm-cache-pusht
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate

export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
# Source h5 already staged + decompressed under datasets/. Frozen encoder is
# the official pretrained PushT LeWM, key-remapped to current code by
# convert_lewm_checkpoint.py (saved under checkpoints/lewm_pusht/).
export PUSHT_DATA=${PUSHT_DATA:-pusht_expert_train.h5}
export LEWM_CKPT=${LEWM_CKPT:-lewm_pusht/weights.pt}
export PUSHT_LATENT_DATA=${PUSHT_LATENT_DATA:-pusht_latent.lance}

mkdir -p logs "$STABLEWM_HOME/datasets" "$STABLEWM_HOME/checkpoints"

echo "[cache-pusht] source=$PUSHT_DATA encoder=$LEWM_CKPT -> $PUSHT_LATENT_DATA"

python scripts/data/cache_latents.py \
  lewm_checkpoint="$LEWM_CKPT" \
  source="$PUSHT_DATA" \
  out_name="$PUSHT_LATENT_DATA" \
  device=cuda \
  batch_size=256
