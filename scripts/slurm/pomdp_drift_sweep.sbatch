#!/bin/bash
#SBATCH -J ldm-drift-sweep
#SBATCH -p gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Stronger drift re-gate. drift_scale=3 gave a real but modest signal (hidden
# residual_ratio 0.0126 vs observed 0.0043; det_R2 0.956 vs 0.989). Sweep larger
# drift to check the signal SCALES (real multimodality) rather than saturating at
# ~1% (which would point to a screen-absorption artifact). Tighter k=64 for more
# local neighborhoods. PASS for the dynamics pipeline = a drift where hidden
# det_R2 drops clearly while observed stays ~0.99.

set -euo pipefail

cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate

export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

for D in 6 10; do
  ds="tworoom_drift${D}.lance"
  echo "==================== drift_scale=${D} ===================="
  echo "[drift${D}] collecting ${ds} ..."
  python scripts/data/collect_tworooms_drift.py \
    cache_dir="$STABLEWM_HOME" num_traj=4000 \
    world.num_envs=10 world.max_episode_steps=100 \
    +drift_scale=${D}.0 +action_noise=2.0 +out_name=${ds}

  echo "[drift${D}] screen HIDDEN ..."
  python scripts/data/multimodality_diagnostic.py \
    --dataset ${ds} --tag drift${D}_hidden \
    --mode dynamics --target-col state --cond-cols state action \
    --k 64 --n-anchors 4000

  echo "[drift${D}] screen OBSERVED ..."
  python scripts/data/multimodality_diagnostic.py \
    --dataset ${ds} --tag drift${D}_observed \
    --mode dynamics --target-col state --cond-cols state drift_state action \
    --k 64 --n-anchors 4000
done

echo "[drift-sweep] DONE. det_r2_mean + residual_ratio by drift:"
echo "   logs/multimodality_drift{6,10}_{hidden,observed}.json"
