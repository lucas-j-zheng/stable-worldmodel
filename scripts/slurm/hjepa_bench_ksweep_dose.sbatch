#!/bin/bash
#SBATCH -J hjepa-bench3
#SBATCH -p gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# H-JEPA E13: bench K-SWEEP + DOSE CURVE -- the "screen predicts the win"
# central figure. x = screened residual/bimodal at (dataset, K); y = MSE
# precision deficit vs mdn/knn at the same (dataset, K).
#
# PRE-REGISTERED (HJEPA_REASONING.md E13):
#   1. K-sweep on mm0: the MSE precision deficit GROWS with K (screen: residual
#      doubles per K-doubling, separation grows). At K=4 deficit smaller than
#      K=8; at K=16 at least as large as K=8 (saturation allowed: T~25).
#   2. Dose curve at K=8: deficit ordered mm0 > p07 > p09 > p10 (monotone in
#      door_prob), tracking the E2/E6 screen ordering.
#   3. knn/mdn precision stays ~flat across doses (they can't be hurt by modes).

set -uo pipefail

cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate

export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

# K-sweep on mm0 (K=8 exists from E9)
for K in 4 16; do
  echo "[bench3] mm0 K=$K ..."
  python scripts/train/proposer_bench.py \
    --dataset tworoom_mm0.lance --horizon $K --epochs 25 \
    --n-anchors 1000 --k-emp 64 --m-samples 32 \
    || echo "[bench3] SKIP mm0 K=$K"
done

# dose curve at K=8 (mm0/p10 exist from E9)
for DS in tworoom_p07 tworoom_p09; do
  echo "[bench3] $DS K=8 ..."
  python scripts/train/proposer_bench.py \
    --dataset ${DS}.lance --horizon 8 --epochs 25 \
    --n-anchors 1000 --k-emp 64 --m-samples 32 \
    || echo "[bench3] SKIP $DS"
done

echo "[bench3] DONE."
