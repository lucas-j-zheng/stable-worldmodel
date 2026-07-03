#!/bin/bash
#SBATCH -J hjepa-hierseeds
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 05:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# H-JEPA E20: seed replication of the closed-loop sample-vs-mean delta at
# offset 12 (delta +6/+8/+8 across three designs, each under the 14 bar).
# PRE-REGISTERED (HJEPA_REASONING.md E20): with seeds 43-45 pooled with the
# seed-42 runs (n=200/arm), count the effect REAL if pooled sampleL-meanL
# >= 6 pts AND every seed's delta >= 0; otherwise report a null closed-loop
# delta (the distributional result stands on its own).

set -uo pipefail
cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate
export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

DET_CKPT=lewm_tworoom/weights_epoch_300.pt

for SEED in 43 44 45; do
  for ARM in sampleL meanL off; do
    echo "[hierseeds] seed=$SEED arm=$ARM ..."
    python scripts/plan/eval_hier.py \
      policy="$DET_CKPT" cache_dir="$LOCAL_DATASET_DIR" \
      eval.goal_offset_steps=12 seed=$SEED \
      plan_config.receding_horizon=2 \
      +hier_arm=$ARM +hier_subgoal_offset=8 +hier_replan_every=10 \
      output.filename="hierseed_${ARM}_s${SEED}_results.txt" \
      || echo "[hierseeds] SKIP $ARM s$SEED"
  done
done

echo "==== SEED RESULTS ===="
for SEED in 43 44 45; do for ARM in sampleL meanL off; do
  head -1 "$STABLEWM_HOME/checkpoints/lewm_tworoom/hierseed_${ARM}_s${SEED}_results.txt" 2>/dev/null || echo "$ARM s$SEED missing"
done; done
