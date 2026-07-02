#!/bin/bash
#SBATCH -J hjepa-bench2
#SBATCH -p gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# H-JEPA E10 + E12 after the E9 confirmation (R11):
#
# E10 LATENT-SPACE proposer bench: same comparison in the frozen LeWM latent
#   space the planner actually uses -- p(z_{t+8} | z_t, goal). 192-d targets;
#   query-space PCA for neighborhoods (mirrors the screen).
#   PRE-REGISTERED: same signature as E9 (MSE precision/coverage collapse on
#   mm0_latent, parity on p10_latent, mdn~knn~diff); weaker separation is
#   acceptable (R3: encoder preserves branching at ~0.8x), but if the gap
#   VANISHES in latent space the level-2 build must use states/goal-relative
#   features, and that is a major finding about the encoder.
#
# E12 SEED REPLICATES of E9 (state space, mm0+p10, torch seeds 1,2): after the
#   +10 seeding collapse, NO gap claim without replicate spread. PRE-REGISTERED:
#   MSE-vs-MDN precision gap on mm0 (~0.24 at seed 0) stable within +/-0.05;
#   p10 parity stable.

set -uo pipefail

cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate

export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

# E10: latent-space bench
for DS in tworoom_mm0_latent tworoom_p10_latent; do
  echo "[bench2] latent $DS ..."
  python scripts/train/proposer_bench.py \
    --dataset ${DS}.lance --state-col latent --horizon 8 --epochs 25 \
    --n-anchors 800 --k-emp 64 --m-samples 32 \
    || echo "[bench2] SKIP latent $DS"
done

# E12: seed replicates in state space
for SEED in 1 2; do
  for DS in tworoom_mm0 tworoom_p10; do
    echo "[bench2] seeds $DS s$SEED ..."
    python scripts/train/proposer_bench.py \
      --dataset ${DS}.lance --horizon 8 --epochs 25 --seed $SEED \
      --n-anchors 1000 --k-emp 64 --m-samples 32 \
      || echo "[bench2] SKIP $DS s$SEED"
  done
done

echo "[bench2] DONE. logs/proposer_bench_*latent*.json + *_s{1,2}.json"
