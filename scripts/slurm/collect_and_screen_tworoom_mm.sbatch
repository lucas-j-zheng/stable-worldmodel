#!/bin/bash
#SBATCH -J ldm-tworoom-mm
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

# 1. Collect the deliberately-multimodal TwoRoom (2 doors, random committed door).
echo "[mm-tworoom] collecting ..."
python scripts/data/collect_tworooms_multimodal.py \
  cache_dir="$STABLEWM_HOME" num_traj=6000 \
  world.num_envs=10 world.max_episode_steps=100

# 2. Screen it vs the greedy-expert TwoRoom (same state-conditioned settings).
#    MILESTONE: residual_bimodal_frac should JUMP (designed hard 2-mode split)
#    where the greedy expert reads low -> validates the screen on real behavioral
#    data AND confirms we finally have a provably-multimodal domain.
echo "[mm-tworoom] screen: multimodal vs greedy ..."
python scripts/data/multimodality_diagnostic.py \
  --dataset tworoom_multimodal.lance --tag tworoom_mm_policy \
  --mode policy
python scripts/data/multimodality_diagnostic.py \
  --dataset tworoom_multimodal.lance --tag tworoom_mm_chunk \
  --mode policy_chunk --cond-from state --chunk 8
python scripts/data/multimodality_diagnostic.py \
  --dataset tworoom_latent_e300.lance --tag tworoom_greedy_chunk \
  --mode policy_chunk --cond-from state --chunk 8

echo "[mm-tworoom] done. compare logs/multimodality_tworoom_{mm,greedy}_*.json"
