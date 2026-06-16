#!/bin/bash
#SBATCH -J ldm-human-screen
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate            # the working 3.10 venv (reads the .lance)

export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
export LEWM_CKPT=${LEWM_CKPT:-lewm_pusht/weights.pt}
mkdir -p logs

# 1. Cache human-PushT latents through the SAME encoder as the scripted expert.
echo "[human] caching latents: pusht_human.lance -> pusht_human_latent.lance"
python scripts/data/cache_latents.py \
  lewm_checkpoint="$LEWM_CKPT" \
  source=pusht_human.lance \
  out_name=pusht_human_latent.lance \
  device=cuda batch_size=256

# 2. The decisive screen. Compare human (multi-demonstrator) vs scripted expert.
#    The ENCODER-FREE state-conditioned action-chunk run (--cond-from state) is
#    the cleanest cross-dataset test -- no domain-gap confound. Also run the
#    latent-conditioned chunk for continuity, and re-run the expert with the
#    same state-conditioned setting for an apples-to-apples baseline.
echo "[human] multimodality screen ..."
python scripts/data/multimodality_diagnostic.py \
  --dataset pusht_human_latent.lance --tag pushthuman_chunk_state \
  --mode policy_chunk --cond-from state --chunk 8
python scripts/data/multimodality_diagnostic.py \
  --dataset pusht_human_latent.lance --tag pushthuman_chunk_latent \
  --mode policy_chunk --cond-from latent --chunk 8

echo "[human] expert baseline (same state-conditioned chunk setting) ..."
python scripts/data/multimodality_diagnostic.py \
  --dataset pusht_latent.lance --tag pushtexpert_chunk_state \
  --mode policy_chunk --cond-from state --chunk 8

echo "[human] done. compare logs/multimodality_pushthuman_chunk_state.json vs"
echo "        logs/multimodality_pushtexpert_chunk_state.json"
