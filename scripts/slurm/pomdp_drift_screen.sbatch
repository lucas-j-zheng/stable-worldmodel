#!/bin/bash
#SBATCH -J ldm-drift-screen
#SBATCH -p gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# P1 GATE #2: does a hidden per-episode DRIFT make the DYNAMICS multimodal
# PERVASIVELY (vs op#1 hidden-doors, which was too sparse)? Cheap, ENCODER-FREE.
#   PASS = hidden det_R2 drops hard (~0-0.3) + residual_bimodal HIGH, while
#          observed (drift in cond) stays det_R2 ~0.99 / residual_bimodal ~0.
#          -> partial obs creates real multimodal dynamics -> greenlight the
#          encoder retrain + diffusion-dynamics-vs-deterministic pipeline.
#   FAIL = hidden ~ observed -> drift didn't register; rethink before any GPU spend.

set -euo pipefail

cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate

export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

# 1. Collect drift POMDP: hidden per-episode drift in {-3,+3} along x.
echo "[drift] collecting tworoom_drift.lance ..."
python scripts/data/collect_tworooms_drift.py \
  cache_dir="$STABLEWM_HOME" num_traj=4000 \
  world.num_envs=10 world.max_episode_steps=100 \
  +drift_scale=3.0 +action_noise=2.0 +out_name=tworoom_drift.lance

# 2a. HIDDEN drift = the POMDP the model sees. p(next_agent_pos | agent_pos, a).
#     Expect det_R2 LOW + residual_bimodal HIGH (two drift branches per step).
echo "[drift] screen HIDDEN (drift not observed) ..."
python scripts/data/multimodality_diagnostic.py \
  --dataset tworoom_drift.lance --tag drift_hidden \
  --mode dynamics --target-col state --cond-cols state action \
  --k 96 --n-anchors 4000

# 2b. OBSERVED drift = full state. p(next_agent_pos | agent_pos, drift_state, a).
#     Expect det_R2 ~0.99 + residual_bimodal ~0 (deterministic once drift known).
echo "[drift] screen OBSERVED (drift in the conditioning) ..."
python scripts/data/multimodality_diagnostic.py \
  --dataset tworoom_drift.lance --tag drift_observed \
  --mode dynamics --target-col state --cond-cols state drift_state action \
  --k 96 --n-anchors 4000

echo "[drift] DONE. Compare residual_bimodality_fraction + det_r2_mean:"
echo "         logs/multimodality_drift_hidden.json    (expect det_R2 LOW, bimodal HIGH)"
echo "         logs/multimodality_drift_observed.json   (expect det_R2 ~0.99, bimodal ~0)"
