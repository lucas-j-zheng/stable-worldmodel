#!/bin/bash
#SBATCH -J ldm-tworoom-mm-lat
#SBATCH -p gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate

export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

# 1. Cache multimodal-TwoRoom latents through the SAME tworoom encoder used for
#    the greedy data -> conditioning on the latent (which encodes agent+target+
#    doors, the full scene the model sees) is the definitive multimodality test
#    and sidesteps the 2-D agent-only `state` column.
echo "[mm-lat] caching multimodal-tworoom latents ..."
python scripts/data/cache_latents.py \
  lewm_checkpoint=lewm_tworoom/weights_epoch_300.pt \
  source=tworoom_multimodal.lance \
  out_name=tworoom_mm_latent.lance \
  device=cuda batch_size=256

# 2. Latent-conditioned action-chunk screen (overfit-guarded), multimodal vs
#    greedy. PREDICTION: greedy stays low (door determined by the image),
#    multimodal JUMPS (door committed at random, invisible to the image) ->
#    confirms designed multimodality in the model's actual conditioning space.
echo "[mm-lat] latent-chunk screen: multimodal vs greedy ..."
python scripts/data/multimodality_diagnostic.py \
  --dataset tworoom_mm_latent.lance --tag tworoom_mm_latentchunk \
  --mode policy_chunk --cond-from latent --chunk 8
python scripts/data/multimodality_diagnostic.py \
  --dataset tworoom_latent_e300.lance --tag tworoom_greedy_latentchunk \
  --mode policy_chunk --cond-from latent --chunk 8

echo "[mm-lat] done. compare residual_bimodality_fraction (mm should >> greedy)."
