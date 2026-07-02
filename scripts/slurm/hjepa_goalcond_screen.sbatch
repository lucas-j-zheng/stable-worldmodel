#!/bin/bash
#SBATCH -J hjepa-goalcond
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# H-JEPA E6: GOAL-CONDITIONED K-step screen -- the decisive re-screen after R2.
#
# R2 falsified the dose prediction: the `none` cell's multimodality is dominated
# by per-episode GOAL variation (cond = state only), not door choice -- p10 read
# the same as mm0. The planner-relevant proposer is goal-conditioned:
#     p(state_{t+K} | state_t, goal)
# Under goal conditioning the spurious goal-branching disappears and what
# remains is ROUTE branching -- the controllable, dose-dependent kind.
#
# PRE-REGISTERED (HJEPA_REASONING.md E6):
#   1. dose RESTORED: mm0/dp05 K=8/none bimodal_frac exceeds p10's by >0.10,
#      with mm0 residual_ratio >= 2x p10's (door branching is real spread).
#   2. rise with K on dp05/mm0 (more windows straddle the door commitment).
#   3. gap_ratio small (<0.3) in the dp05 K=8/none cell (mean-in-gap mechanism);
#      ~1 on p10.
#   4. If mm0 == p10 == floor here too: NO controllable route branching at
#      K-scale in TwoRoom => kill the domain, escalate to mazes.

set -uo pipefail

cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate

export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

# 0. probe: what columns do the datasets actually have (goal col name differs
#    across dataset generations)
python - << 'PY'
import stable_worldmodel as swm
for name in ['tworoom_mm_dp05.lance', 'tworoom_mm0.lance', 'tworoom_p10.lance']:
    try:
        print('[cols]', name, sorted(swm.data.load_dataset(name).column_names))
    except Exception as e:
        print('[cols]', name, 'ERR', repr(e))
PY

GCANDS="goal_state variation_target_position goal target_position goal_pos"

run_cell () {  # DS TAG K COARSE
  local DS=$1 TAG=$2 K=$3 COARSE=$4 GCOL
  for GCOL in $GCANDS; do
    if python scripts/data/multimodality_diagnostic.py \
        --dataset "$DS" --tag "$TAG" \
        --mode dynamics_k --horizon "$K" --coarse-action "$COARSE" \
        --target-col state --cond-cols state "$GCOL" \
        --k 96 --n-anchors 4000; then
      echo "[hjepa-goal] $TAG OK (goal col = $GCOL)"
      return 0
    fi
  done
  echo "[hjepa-goal] SKIP $TAG (no goal column worked)"
  return 0
}

# dp05: full K sweep, three abstractions incl. the new `dir`
for K in 1 2 4 8; do
  for COARSE in none dir sum; do
    run_cell tworoom_mm_dp05.lance "hjepa_goal_dp05_k${K}_${COARSE}" $K $COARSE
  done
done

# dose endpoints: mm0 vs p10 (the controlled contrast), K in {1,8}
for DS in tworoom_mm0 tworoom_p10; do
  for K in 1 8; do
    for COARSE in none sum; do
      run_cell ${DS}.lance "hjepa_goal_${DS}_k${K}_${COARSE}" $K $COARSE
    done
  done
done

echo "[hjepa-goal] DONE. JSONs: logs/multimodality_hjepa_goal_*.json"
