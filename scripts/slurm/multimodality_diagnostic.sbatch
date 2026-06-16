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
# Full 2x2: dynamics-multimodality (what a diffusion *dynamics* model exploits)
# vs policy-multimodality (what a diffusion *policy/proposal* exploits), on the
# near-deterministic reference (TwoRoom) and the E6 candidate (PushT). The
# detrended residual_bimodality_fraction is the comparable signal.
DOMAINS=("tworoom:tworoom_latent_e300.lance" "pusht:pusht_latent.lance")
for entry in "${DOMAINS[@]}"; do
  dom="${entry%%:*}"; ds="${entry##*:}"
  for mode in dynamics policy; do
    echo "[multimodal] ${dom} / ${mode} ..."
    python scripts/data/multimodality_diagnostic.py \
      --dataset "$ds" --tag "${dom}_${mode}" --mode "$mode"
  done
done

echo "[multimodal] done. JSONs: logs/multimodality_{tworoom,pusht}_{dynamics,policy}.json"
