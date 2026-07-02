#!/bin/bash
#SBATCH -J hjepa-maze-p4
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# H-JEPA E11: maze retry with a PERSISTENT random walk (hold action 4 steps)
# and T=128, K up to 32. R10 hit pre-registered risk 4: a per-step random walk
# diffuses ~sqrt(K) and never straddles junctions by K=16 (res_ratio 0.05,
# bimodal ~0). A persistent walk covers ~persist*sqrt(K/persist) distance.
# PRE-REGISTERED: `none` bimodal_frac rises with K and exceeds 0.15 by K=16-32
# with res_ratio >= 0.15 IF corridors/junctions are reachable at that range;
# full stays the composed-determinism control. If still flat: medium maze
# junction spacing simply exceeds random-walk reach -> switch to goal-directed
# expert data (ogbench experts) rather than bigger K.

set -uo pipefail

cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate

export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

DS=pointmaze_medium_rand_p4_T128.lance

if [ ! -d "$STABLEWM_HOME/datasets/${DS}" ]; then
  python scripts/data/collect_maze_screen.py \
    --episodes 2000 --maze-type medium --steps 128 --num-envs 8 \
    --persist 4 --out $DS
else
  echo "[maze-p4] dataset exists, skipping collection"
fi

for K in 4 8 16 32; do
  for COARSE in full sum none; do
    echo "[maze-p4] K=$K coarse=$COARSE ..."
    python scripts/data/multimodality_diagnostic.py \
      --dataset $DS --tag "hjepa_mazep4_k${K}_${COARSE}" \
      --mode dynamics_k --horizon $K --coarse-action $COARSE \
      --target-col observation --cond-cols observation \
      --k 96 --n-anchors 4000 \
      || echo "[maze-p4] SKIP K=$K $COARSE"
  done
done

echo "[maze-p4] DONE. logs/multimodality_hjepa_mazep4_*.json"
