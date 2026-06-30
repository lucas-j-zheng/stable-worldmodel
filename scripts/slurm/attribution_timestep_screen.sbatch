#!/bin/bash
#SBATCH -J ldm-attrib-tstep
#SBATCH -p gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# ATTRIBUTION PUZZLE (audit D2). The diffusion-policy +10 tracks the door_prob
# dose, yet under the policy's ACTUAL conditioning (policy_goal = current + the
# +8 future position as goal) the aggregate screen reads FLAT ~0.14. The dose
# only appears under policy_target (final destination). Those two modes differ in
# BOTH goal-type AND timestep-range -- a confound.
#
# This screen isolates the TIMESTEP axis, holding goal-type fixed at the policy's
# +8 (policy_goal). Hypothesis: the +8 goal leaks the door only LATE (once the
# agent nears it); EARLY (route still uncommitted) the same (state, +8 goal) maps
# to two action chunks -> multimodal. If early >> late on the multimodal data,
# the policy DOES face real conditional multimodality at decision time and the
# aggregate +8 screen merely washes it out -> attribution holds, headline clean.
# If early ~ late ~ flat, the +10 is NOT test-time conditional multimodality (it
# comes from training on multimodal demos / the marginal) -> reframe the headline.
#
# Encoder-free (state+action). ~3 min.

set -euo pipefail
cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate
export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

run () {  # ds mode tag frac_lo frac_hi
  local ds="$1" mode="$2" tag="$3" lo="$4" hi="$5"
  echo "==== ${tag}: ${ds} mode=${mode} window=[${lo},${hi}) ===="
  python scripts/data/multimodality_diagnostic.py \
    --dataset "${ds}.lance" --tag "attrib_${tag}" \
    --mode "${mode}" --chunk 8 --goal-offset 8 \
    --frac-lo "${lo}" --frac-hi "${hi}" \
    --k 96 --n-anchors 4000 2>&1 \
    | grep -iE "RESULT|mode=policy|det_r2_mean|residual_bimodality_fraction|residual_ratio|ratio_cond|pairs,|absent|Error|WARNING" \
    || echo "[attrib] ${tag} FAILED (missing dataset?)"
}

# --- KEY DECOMPOSITION: door_prob 0.5 (MULTIMODAL), policy_goal (+8), by timestep
run tworoom_mm0 policy_goal mm0_goal_early 0.0  0.33
run tworoom_mm0 policy_goal mm0_goal_mid   0.33 0.66
run tworoom_mm0 policy_goal mm0_goal_late  0.66 1.0
run tworoom_mm0 policy_goal mm0_goal_full  0.0  1.0   # reproduce flat ~0.14 baseline

# --- REFERENCE: destination goal (no leak) on the same data, early vs full
run tworoom_mm0 policy_target mm0_tgt_early 0.0 0.33
run tworoom_mm0 policy_target mm0_tgt_full  0.0 1.0

# --- UNIMODAL CONTRAST: door_prob 1.0 (always door 0), policy_goal early
# If early mm is HIGH on mm0 but LOW here, the timestep-resolved dose is real.
run tworoom_p10 policy_goal p10_goal_early 0.0  0.33
run tworoom_p10 policy_goal p10_goal_full  0.0  1.0

echo "[attrib] DONE. Compare residual_bimodality_fraction / residual_ratio:"
echo "   mm0 policy_goal early vs mid vs late vs full  (does mm live early?)"
echo "   mm0 early: policy_goal(+8) vs policy_target(dest)  (does goal-type matter once windowed?)"
echo "   mm0 vs p10 policy_goal early  (is the early dose multimodality-specific?)"
echo "   JSON: logs/multimodality_attrib_*.json"
