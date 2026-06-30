#!/bin/bash
#SBATCH -J ldm-attrib-latgoal
#SBATCH -p gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# ATTRIBUTION, step 2 (audit D2). The state-goal screen (job 3575246) showed that
# under the raw +8 STATE goal, p(action|history,goal) is ~unimodal at EVERY
# timestep and mm0 ~ p10 (multimodal data indistinguishable from unimodal). But
# the closed-loop policy conditions on the +8 LATENT (encoded image), which is
# LOSSIER than the raw state and may preserve route ambiguity the state resolves.
# This is the conditioning-MATCHED screen: condition on (latent_t, latent_{t+8}),
# exactly what the diffusion/TMSE policies see.
#
# DECIDES the headline wording:
#   latent-goal mm0 ~ p10 (both low)  -> the +10 is NOT test-time conditional mm;
#       it is a TRAINING-MARGINAL effect (MSE averages the dataset's route modes)
#       -> reframe "advantage requires multimodality".
#   latent-goal mm0 >> p10            -> the lossy latent preserves the dose; the
#       advantage IS conditional-multimodality under the policy's real conditioning
#       -> headline survives.
# Encoder-free of NEW compute (latents already cached). ~3 min.

set -euo pipefail
cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate
export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

run () {  # ds tag frac_lo frac_hi
  local ds="$1" tag="$2" lo="$3" hi="$4"
  echo "==== ${tag}: ${ds} policy_goal cond_from=latent window=[${lo},${hi}) ===="
  python scripts/data/multimodality_diagnostic.py \
    --dataset "${ds}.lance" --tag "attriblat_${tag}" \
    --mode policy_goal --cond-from latent --chunk 8 --goal-offset 8 \
    --frac-lo "${lo}" --frac-hi "${hi}" \
    --k 96 --n-anchors 4000 2>&1 \
    | grep -iE "RESULT|mode=policy|det_r2_mean|residual_bimodality_fraction|residual_ratio|ratio_cond|PCA|pairs,|absent|Error|WARNING" \
    || echo "[attriblat] ${tag} FAILED (missing dataset?)"
}

# MULTIMODAL (door_prob 0.5) vs UNIMODAL (door_prob 1.0), latent +8 goal.
run tworoom_mm0_latent mm0_latgoal_full  0.0 1.0
run tworoom_mm0_latent mm0_latgoal_early 0.0 0.33
run tworoom_p10_latent p10_latgoal_full  0.0 1.0
run tworoom_p10_latent p10_latgoal_early 0.0 0.33

echo "[attriblat] DONE. Compare residual_bimodality_fraction / residual_ratio:"
echo "   mm0_latgoal vs p10_latgoal  (does the LATENT goal preserve the dose the STATE goal killed?)"
echo "   JSON: logs/multimodality_attriblat_*.json"
