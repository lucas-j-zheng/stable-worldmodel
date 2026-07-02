#!/bin/bash
#SBATCH -J hjepa-propbench
#SBATCH -p gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# H-JEPA E9 / RUNG 3: proposer bench on the E6-greenlit conditional
# p(s_{t+8} | s_t, goal). MSE vs gauss-NLL vs MDN vs kNN-retrieval vs diffusion,
# evaluated distributionally per anchor vs the held-out empirical conditional.
# Dose contrast: mm0 (multimodal demonstrator) vs p10 (unimodal).
#
# PRE-REGISTERED (HJEPA_REASONING.md E9):
#   1. mm0: MSE precision LOW (mean lands between the doors), meangap HIGH;
#      generative heads (mdn/knn/diff) precision HIGH, meangap LOW.
#   2. p10: parity across heads (nothing to win) -- the seeded-2x2 lesson says
#      claim NOTHING without the unimodal control.
#   3. mdn ~ knn ~ diff on mm0 (2 hard modes: any multimodal head suffices;
#      diffusion is NOT special here) -- the "which generative model" refinement.
#   4. gauss ~ mse (calibrated but unimodal does not fix mode-averaging).
#   5. mode_coverage: generative heads > 0.8, mse ~ 0.5 (mean hits neither/one).

set -uo pipefail

cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate

export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

for DS in tworoom_mm0 tworoom_p10 tworoom_mm_dp05; do
  echo "[propbench] $DS ..."
  python scripts/train/proposer_bench.py \
    --dataset ${DS}.lance --horizon 8 --epochs 25 \
    --n-anchors 1000 --k-emp 64 --m-samples 32 \
    || echo "[propbench] SKIP $DS"
done

echo "[propbench] DONE. logs/proposer_bench_*.json"
