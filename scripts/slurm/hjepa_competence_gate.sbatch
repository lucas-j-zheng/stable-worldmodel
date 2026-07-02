#!/bin/bash
#SBATCH -J hjepa-compgate
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 05:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# H-JEPA E17 / RUNG-5 PREREQUISITE: level-1 competence gate (review C8).
# Flat LeWM CEM (deterministic predictor, the project's best planner) on
# multimodal TwoRoom, goal offset swept {4, 8, 12, 16, 20} (episodes T~25 cap
# usable offsets at ~20). Success-vs-offset curve sets the subgoal step K* the
# hierarchy can rely on and the flat baseline the hierarchy must beat.
#
# PRE-REGISTERED (HJEPA_REASONING.md E17):
#   1. success decreases monotonically in offset (noise +/-7 at n=50);
#   2. K* = largest offset with success >= 40%: expect K* in [8, 16] (flat
#      F=8 CEM was built for ~8-step goals; 58% at the old settings);
#   3. if success at offset 20 is STILL >= 40%, flat planning has no
#      long-horizon deficit on this domain at reachable offsets -> rung 5's
#      capability payoff needs longer tasks (cross-episode goals), though the
#      thesis payoff (kNN vs MSE proposer at fixed K) is unaffected.

set -uo pipefail

cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate
export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

DET_CKPT=${DET_CKPT:-lewm_tworoom/weights_epoch_300.pt}

for OFF in 4 8 12 16 20; do
  echo "[compgate] offset=$OFF ..."
  python scripts/plan/eval_wm.py --config-name tworoom_mm_diffusion \
    policy="$DET_CKPT" cache_dir="$LOCAL_DATASET_DIR" \
    eval.goal_offset_steps=$OFF \
    output.filename="compgate_off${OFF}_results.txt" \
    || echo "[compgate] SKIP offset $OFF"
done

echo "==== COMPETENCE CURVE ===="
for OFF in 4 8 12 16 20; do
  echo "-- offset $OFF:"
  tail -3 "$STABLEWM_HOME/checkpoints/lewm_tworoom/compgate_off${OFF}_results.txt" 2>/dev/null || echo "  (missing)"
done
