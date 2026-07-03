#!/bin/bash
#SBATCH -J hjepa-hier2
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 05:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# H-JEPA E18 v2 (after R23): fix the two v1 harness flaws.
#   (a) cadence: receding_horizon 5 -> 2 (inner replans every 10 env steps,
#       matched to K=8 subgoals) -- applied to ALL arms incl. flat, so the
#       comparison stays compute-matched;
#   (b) final-goal handoff: aim at the TRUE goal once it is within ~1.25x the
#       median 8-step hop (proposer arms previously never targeted the goal).
#   Subgoal refresh every 10 steps (aligned to inner cadence).
#
# PRE-REGISTERED (HJEPA_REASONING.md E18v2):
#   1. flat(receding2) may itself improve vs flat(receding5) -- report it;
#      the hierarchy must beat the NEW flat.
#   2. sample > mean at offset 12 (>14 points to count);
#   3. sample vs flat: expected positive at 12 but registered as OPEN -- if
#      flat(receding2) already solves offset 12, the capability headroom moves
#      to 16/20 and the thesis test lives in sample-vs-mean only.

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
    echo "[hier2] offset=$OFF arm=$ARM ..."
    python scripts/plan/eval_hier.py \
      policy="$DET_CKPT" cache_dir="$LOCAL_DATASET_DIR" \
      eval.goal_offset_steps=$OFF \
      plan_config.receding_horizon=2 \
      +hier_arm=$ARM +hier_subgoal_offset=8 +hier_replan_every=10 \
      output.filename="hier2_${ARM}_off${OFF}_results.txt" \
      || echo "[hier2] SKIP $ARM off$OFF"
  done
done

echo "==== HIER2 RESULTS ===="
for OFF in 12 16; do for ARM in sample mean off; do
  head -1 "$STABLEWM_HOME/checkpoints/lewm_tworoom/hier2_${ARM}_off${OFF}_results.txt" 2>/dev/null || echo "$ARM off$OFF missing"
done; done
