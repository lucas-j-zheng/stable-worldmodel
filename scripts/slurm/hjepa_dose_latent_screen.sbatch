#!/bin/bash
#SBATCH -J hjepa-dose-latent
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# H-JEPA RUNG 1.5b + RUNG 2 (plan review C2/C7 + design-doc rung 2), zero new
# collection -- all datasets already cached in scratch from the policy arc.
#
# 1.5b DOSE GRID (encoder-free, state target): K x coarse on door_prob
#      0.5 (mm0) / 0.7 (p07) / 0.9 (p09) / 1.0 (p10).
#      PREDICTION (pre-registered): at K=8/none, residual_bimodal_frac decreases
#      monotonically in door_prob and is ~0 at p10; at K=1 all cells ~0.
#      This is the x-axis of the "screen predicts the win" dose figure.
#
# 2    LATENT CONFIRMATION: same screen with --target-col latent on the cached
#      LeWM latents (mm0_latent, p10_latent contrast). Does the K-step branching
#      survive the SIGReg encoder? PREDICTION: latent-target bimodal_frac tracks
#      the state-target cell (same K/coarse) within ~2x; p10 stays at floor.
#
# Cells are guarded (|| echo SKIP) so one missing column doesn't kill the sweep.

set -uo pipefail

cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate

export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

# --- 1.5b: dose grid, encoder-free state target -----------------------------
for DS in tworoom_mm0 tworoom_p07 tworoom_p09 tworoom_p10; do
  for K in 1 8; do
    for COARSE in full sum none; do
      echo "[hjepa-dose] $DS K=$K coarse=$COARSE ..."
      python scripts/data/multimodality_diagnostic.py \
        --dataset ${DS}.lance --tag "hjepa_dose_${DS}_k${K}_${COARSE}" \
        --mode dynamics_k --horizon $K --coarse-action $COARSE \
        --target-col state --cond-cols state \
        --k 96 --n-anchors 4000 \
        || echo "[hjepa-dose] SKIP $DS K=$K $COARSE (see error above)"
    done
  done
done

# --- 2: latent-target confirmation (rung 2) ---------------------------------
for DS in tworoom_mm0_latent tworoom_p10_latent; do
  for K in 1 8; do
    for COARSE in full sum none; do
      echo "[hjepa-lat] $DS K=$K coarse=$COARSE ..."
      python scripts/data/multimodality_diagnostic.py \
        --dataset ${DS}.lance --tag "hjepa_lat_${DS}_k${K}_${COARSE}" \
        --mode dynamics_k --horizon $K --coarse-action $COARSE \
        --target-col latent \
        --k 96 --n-anchors 4000 \
        || echo "[hjepa-lat] SKIP $DS K=$K $COARSE (see error above)"
    done
  done
done

echo "[hjepa-dose-latent] DONE. JSONs: logs/multimodality_hjepa_{dose,lat}_*.json"
