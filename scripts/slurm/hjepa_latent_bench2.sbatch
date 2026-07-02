#!/bin/bash
#SBATCH -J hjepa-latbench2
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# H-JEPA E16: latent bench, round 2 (after R19).
# R19: gap present in latent space via kNN (mse 0.53 vs knn 0.84) but the
# PARAMETRIC generative heads degraded at 192-d (mdn 0.62) and the small DDPM
# failed outright (precision 0.000, loss still falling at 25 epochs). Also the
# p10_latent CONTROL never ran (gpu-debug 1h timeout).
#
# Fix: hidden 512, epochs 75, non-debug partition. PRE-REGISTERED:
#   1. p10_latent control: mse ~ knn parity (no modes, no gap) -- REQUIRED for
#      the latent dose contrast claim.
#   2. tuned mdn/diff on mm0_latent close most of the gap to knn
#      (precision >= 0.75); if diff still ~0, the small-MLP DDPM is inadequate
#      at 192-d and the paper reports kNN/MDN as the latent-space generative
#      representatives (claim unchanged: law is about the objective family).

set -uo pipefail

cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate

export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

for DS in tworoom_p10_latent tworoom_mm0_latent; do
  echo "[latbench2] $DS ..."
  python scripts/train/proposer_bench.py \
    --dataset ${DS}.lance --state-col latent --goal-from final \
    --horizon 8 --epochs 75 --hidden 512 \
    --n-anchors 800 --k-emp 64 --m-samples 32 \
    --out logs/proposer_bench_${DS}_k8_latent_v2.json \
    || echo "[latbench2] SKIP $DS"
done

echo "[latbench2] DONE."
