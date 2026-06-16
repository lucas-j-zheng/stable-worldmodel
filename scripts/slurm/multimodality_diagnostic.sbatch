#!/bin/bash
#SBATCH -J ldm-multimodal
#SBATCH -p gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate

export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}

mkdir -p logs

# Pure numpy/sklearn (no GPU use) -- on gpu-debug only to dodge the CPU-QOS wall.
# TwoRoom = the near-deterministic negative reference; PushT = the multimodal
# candidate E6 is testing. Same metric on both makes E6 interpretable.
echo "[multimodal] TwoRoom (e300 latents) ..."
python scripts/data/multimodality_diagnostic.py \
  --dataset tworoom_latent_e300.lance --tag tworoom \
  --cond-cols state action

echo "[multimodal] PushT (pretrained-encoder latents) ..."
python scripts/data/multimodality_diagnostic.py \
  --dataset pusht_latent.lance --tag pusht \
  --cond-cols state action

echo "[multimodal] done. JSONs in logs/multimodality_{tworoom,pusht}.json"
