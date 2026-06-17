#!/bin/bash
#SBATCH -J ldm-human-clean
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

python -c "import zarr" 2>/dev/null || pip install -q "zarr<3" "numcodecs"

# 1. Human full 5-D state (agent+block) + action from the DP zarr.
echo "[clean] extracting human full-state from DP zarr ..."
python scripts/data/convert_pusht_zarr_state.py --out pusht_human_state.lance

# 2. MATCHED, overfit-guarded comparison (k auto-bumped to >> params).
#    (a) clean encoder-free state-conditioned action-chunk: human 5-D vs expert 7-D
#    (b) latent-conditioned action-chunk, now overfit-guarded, both datasets
echo "[clean] state-conditioned chunk: human (5-D) vs expert (7-D) ..."
python scripts/data/multimodality_diagnostic.py \
  --dataset pusht_human_state.lance --tag human_state5_chunk \
  --mode policy_chunk --cond-from state --chunk 8
python scripts/data/multimodality_diagnostic.py \
  --dataset pusht_latent.lance --tag expert_state7_chunk \
  --mode policy_chunk --cond-from state --chunk 8

echo "[clean] latent-conditioned chunk (overfit-guarded): human vs expert ..."
python scripts/data/multimodality_diagnostic.py \
  --dataset pusht_human_latent.lance --tag human_latent_chunk_guarded \
  --mode policy_chunk --cond-from latent --chunk 8
python scripts/data/multimodality_diagnostic.py \
  --dataset pusht_latent.lance --tag expert_latent_chunk_guarded \
  --mode policy_chunk --cond-from latent --chunk 8

echo "[clean] done. compare residual_bimodality_fraction across the 4 JSONs."
