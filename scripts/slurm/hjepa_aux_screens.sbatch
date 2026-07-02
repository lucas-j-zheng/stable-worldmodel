#!/bin/bash
#SBATCH -J hjepa-aux
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# H-JEPA E8 (aux, runs in the E6 shadow):
#
# (a) POLICY-PUZZLE RETRO-TEST with the GMM soft-mode metric. The 06-30 verdict
#     ("matched conditional is unimodal, bimodal ~0.006") used Sarle BC, which
#     is blind to SOFT overlapping modes -- the design doc's own hypothesis
#     (S6). gmm2_frac has power there (self-test: trimodal 0.952 vs BC 0.730).
#     Re-run the conditioning-MATCHED policy screen (policy_goal, latent goal)
#     on mm0_latent vs p10_latent.
#     PRE-REGISTERED: if soft modes exist, gmm2_frac(mm0) - gmm2_frac(p10) > 0.1
#     while BC stays ~0.006 for both. If gmm2 also reads equal/floor, hypothesis
#     (a) of the 06-30 puzzle is dead too and the mechanism is elsewhere
#     (training-time corruption / calibration).
#
# (b) SEED-NOISE CALIBRATION for the screen: rerun key K=8 cells at seeds 1,2
#     (seed 0 exists). Gives the run-to-run sigma of bimodal_frac needed to
#     judge dose differences (e.g. sum-cell 0.155 vs 0.124).

set -uo pipefail

cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate

export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

# (a) policy-puzzle retro-test, conditioning-matched (mirrors job 3575527)
for DS in tworoom_mm0_latent tworoom_p10_latent; do
  for WIN in "0.0 1.0 full" "0.0 0.33 early"; do
    set -- $WIN
    echo "[hjepa-aux] policy retro $DS window=$3 ..."
    python scripts/data/multimodality_diagnostic.py \
      --dataset ${DS}.lance --tag "hjepa_polretro_${DS}_$3" \
      --mode policy_goal --cond-from latent --chunk 8 --goal-offset 8 \
      --frac-lo $1 --frac-hi $2 \
      --k 96 --n-anchors 4000 \
      || echo "[hjepa-aux] SKIP $DS $3"
  done
done

# (b) seed calibration on the K=8 decision cells
for SEED in 1 2; do
  for COARSE in sum none; do
    python scripts/data/multimodality_diagnostic.py \
      --dataset tworoom_mm_dp05.lance --tag "hjepa_seed_dp05_k8_${COARSE}_s${SEED}" \
      --mode dynamics_k --horizon 8 --coarse-action $COARSE \
      --target-col state --cond-cols state \
      --k 96 --n-anchors 4000 --seed $SEED \
      || echo "[hjepa-aux] SKIP seed cell dp05 $COARSE s$SEED"
  done
  for DS in tworoom_mm0 tworoom_p10; do
    python scripts/data/multimodality_diagnostic.py \
      --dataset ${DS}.lance --tag "hjepa_seed_${DS}_k8_sum_s${SEED}" \
      --mode dynamics_k --horizon 8 --coarse-action sum \
      --target-col state --cond-cols state \
      --k 96 --n-anchors 4000 --seed $SEED \
      || echo "[hjepa-aux] SKIP seed cell $DS s$SEED"
  done
done

echo "[hjepa-aux] DONE. JSONs: logs/multimodality_hjepa_{polretro,seed}_*.json"
