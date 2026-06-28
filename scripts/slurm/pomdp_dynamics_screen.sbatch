#!/bin/bash
#SBATCH -J ldm-pomdp-screen
#SBATCH -p gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# P1 GATE: does partial observability make the DYNAMICS multimodal?
# Cheap, ENCODER-FREE screen that must pass BEFORE any encoder retrain / diffusion
# training. Collect TwoRoom with randomized HIDDEN doors, then screen next-state
# dynamics with doors HIDDEN vs OBSERVED.
#   PASS  = hidden residual_bimodal_frac >> observed (~0). Partial obs creates
#           multimodal dynamics -> greenlight the full diffusion-dynamics pipeline.
#   FAIL  = hidden ~ observed ~ low. History/obs already determines the future ->
#           stop before burning GPU-days.

set -euo pipefail

cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate

export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

# 0. Sanity: the screen still flags synthetic bimodal and clears unimodal.
echo "[pomdp] self-test (expect unimodal~0, bi/tri HIGH) ..."
python scripts/data/multimodality_diagnostic.py --self-test

# 1. Collect POMDP TwoRoom: random hidden doors per episode, exploratory policy.
echo "[pomdp] collecting tworoom_pomdp.lance ..."
python scripts/data/collect_tworooms_pomdp.py \
  cache_dir="$STABLEWM_HOME" num_traj=4000 \
  world.num_envs=10 world.max_episode_steps=100 \
  action_noise=2.0 out_name=tworoom_pomdp.lance

# 2a. HIDDEN doors = the POMDP the model sees. p(next_agent_pos | agent_pos, a).
#     Expect residual_bimodal HIGH (pass vs bounce at the wall under hidden doors).
echo "[pomdp] screen HIDDEN (doors not observed) ..."
python scripts/data/multimodality_diagnostic.py \
  --dataset tworoom_pomdp.lance --tag pomdp_hidden \
  --mode dynamics --target-col state --cond-cols state action \
  --k 96 --n-anchors 4000

# 2b. OBSERVED doors = full state. p(next_agent_pos | agent_pos, door_state, a).
#     Expect residual_bimodal ~0 (deterministic once the door layout is known).
echo "[pomdp] screen OBSERVED (doors in the conditioning) ..."
python scripts/data/multimodality_diagnostic.py \
  --dataset tworoom_pomdp.lance --tag pomdp_observed \
  --mode dynamics --target-col state --cond-cols state door_state action \
  --k 96 --n-anchors 4000

echo "[pomdp] DONE. Compare residual_bimodality_fraction:"
echo "         logs/multimodality_pomdp_hidden.json   (expect HIGH)"
echo "         logs/multimodality_pomdp_observed.json  (expect ~0)"
