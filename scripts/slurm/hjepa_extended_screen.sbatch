#!/bin/bash
#SBATCH -J hjepa-extended
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# H-JEPA E5: EXTENDED rung-1 screen with the upgraded instrument (review B3/B9/C1)
# on the dp05 dataset collected by rung 1 (job 3611937). Adds:
#   - coarseness DOSE: full / sum / sumhalf / dir / none (monotone-in-bits check;
#     disambiguates Sigma-a-sufficiency from no-branching)
#   - stride=K decorrelated robustness cells (window-overlap check)
#   - GMM2 soft-mode fraction + mean-gap ratio (the infeasible-average-mechanism
#     metric) -- emitted automatically by the upgraded diagnostic.
# PRE-REGISTERED (HJEPA_REASONING.md E4/E5): monotone bimodal_frac in coarseness;
# stride=K within +/-30% of stride=1; gap_ratio <= 0.2 where bimodal_frac is high.

set -uo pipefail

cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate

export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

# self-test first: new metrics must reproduce the validated control pattern on
# the cluster env too
python scripts/data/multimodality_diagnostic.py --self-test --n-anchors 600 --k 96

DS=tworoom_mm_dp05.lance

# coarseness x K grid
for K in 1 2 4 8; do
  for COARSE in full sum sumhalf dir none; do
    echo "[hjepa-ext] K=$K coarse=$COARSE ..."
    python scripts/data/multimodality_diagnostic.py \
      --dataset $DS --tag "hjepa_ext_k${K}_${COARSE}" \
      --mode dynamics_k --horizon $K --coarse-action $COARSE \
      --target-col state --cond-cols state \
      --k 96 --n-anchors 4000 \
      || echo "[hjepa-ext] SKIP K=$K $COARSE"
  done
done

# decorrelated robustness: non-overlapping windows at K=8
for COARSE in sum none; do
  echo "[hjepa-ext] stride=8 K=8 coarse=$COARSE ..."
  python scripts/data/multimodality_diagnostic.py \
    --dataset $DS --tag "hjepa_ext_k8_${COARSE}_stride8" \
    --mode dynamics_k --horizon 8 --coarse-action $COARSE --stride 8 \
    --target-col state --cond-cols state \
    --k 96 --n-anchors 4000 \
    || echo "[hjepa-ext] SKIP stride cell $COARSE"
done

echo "[hjepa-ext] DONE. JSONs: logs/multimodality_hjepa_ext_*.json"
