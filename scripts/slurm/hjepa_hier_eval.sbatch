#!/bin/bash
#SBATCH -J hjepa-hier
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 05:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# H-JEPA E18 / RUNG 5 (minimal instantiation): closed-loop subgoal proposing.
# Level-1 = stock deterministic-LeWM CEM (untouched). Every 25 env steps the
# wrapper swaps the goal image for a K=8 subgoal frame from a kNN bank over
# (state_t, goal_state) -- the E9-validated conditional. Arms:
#   off    = flat baseline (E17 numbers: 28% @12, 10% @16)
#   mean   = neighbors' MEAN future state, snapped to nearest real frame
#            (conservative-strong MSE analog)
#   sample = ONE random neighbor's future frame (generative sampler)
#
# PRE-REGISTERED (HJEPA_REASONING.md E18):
#   1. sample > mean at offsets {12,16} on mm0 (the R11 precision gap must
#      surface closed-loop; mean subgoals snap arbitrarily between routes /
#      to low-progress frames);
#   2. sample > off (flat) at offsets {12,16}: K=8 subgoals sit inside the
#      competence radius (E17: 52% at 8) so guided planning should beat flat
#      28%/10%;
#   3. n=50 single eval-seed first; +/-7 noise bar from the policy arc --
#      only deltas > ~14 count before seed replicates.

set -uo pipefail

cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate
export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

DET_CKPT=${DET_CKPT:-lewm_tworoom/weights_epoch_300.pt}

for OFF in 12 16; do
  for ARM in sample mean off; do
    echo "[hier] offset=$OFF arm=$ARM ..."
    python scripts/plan/eval_hier.py \
      policy="$DET_CKPT" cache_dir="$LOCAL_DATASET_DIR" \
      eval.goal_offset_steps=$OFF \
      +hier_arm=$ARM +hier_subgoal_offset=8 +hier_replan_every=25 \
      output.filename="hier_${ARM}_off${OFF}_results.txt" \
      || echo "[hier] SKIP $ARM off$OFF"
  done
done

echo "==== HIER RESULTS ===="
for OFF in 12 16; do for ARM in sample mean off; do
  head -1 "$STABLEWM_HOME/checkpoints/lewm_tworoom/hier_${ARM}_off${OFF}_results.txt" 2>/dev/null || echo "$ARM off$OFF missing"
done; done
