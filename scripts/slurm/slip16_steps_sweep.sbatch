#!/bin/bash
#SBATCH -J ldm-slip16-steps
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate
export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

# ARC 3 stage 3 (afterok slip_dynamics_cell TAG=slip16): the falsification test.
# Closed-loop D-MPC on the slip16 env, sweeping DDIM sampling steps for the
# v-diffusion dynamics model: @1 (one-shot readout), @5, @20 (full sampling).
# TMSE control @1 only (deterministic single forward by construction).
# 3 train seeds x 2 eval seeds x n=50 = n=100/cell.
#
# PRE-REGISTERED PREDICTIONS:
#   P1 (sampling penalty is UNIVERSAL): v@1 >= v@20 even at slip16 -> the
#       penalty survives extreme stochasticity; receding-horizon replanning
#       makes the conditional MEAN sufficient regardless of aleatoric spread.
#   P2 (sampling finally PAYS): v@20 > v@1 at slip16 -> BREAKS the universal
#       claim; bounds it to low/moderate stochasticity. Most interesting.
#   P3 (diff vs MSE edge): if the diff-over-tmse gap GROWS vs slip8 (+6), the
#       edge is stochasticity-driven after all (contra the falsified P1-arc
#       interaction); if flat, the training-objective story stands at 2x dose.
SLIP=16
TAG=slip16
EP=${EP:-100}

for s in 1 2 3; do
  # v-diffusion: full step sweep
  name="dyn_diff_${TAG}_ps${s}"
  for st in 1 5 20; do
    for es in 42 123; do
      echo "[slip16-steps] ==== EVAL $name steps=$st (n=50, eval seed $es) ===="
      python scripts/plan/eval_wm.py --config-name tworoom_slip_diffusion \
        policy=${name}/weights_epoch_$EP.pt cache_dir="$LOCAL_DATASET_DIR" \
        seed=$es world.slip_scale=$SLIP eval.dataset_name=tworoom_${TAG}.lance \
        "+model_overrides={num_inference_steps:$st}" 2>&1 \
        | grep -iE "success_rate" | tail -1
    done
  done
  # TMSE control: single forward only
  tname="dyn_tmse_${TAG}_ps${s}"
  for es in 42 123; do
    echo "[slip16-steps] ==== EVAL $tname (TMSE, n=50, eval seed $es) ===="
    python scripts/plan/eval_wm.py --config-name tworoom_slip_diffusion \
      policy=${tname}/weights_epoch_$EP.pt cache_dir="$LOCAL_DATASET_DIR" \
      seed=$es world.slip_scale=$SLIP eval.dataset_name=tworoom_${TAG}.lance \
      2>&1 | grep -iE "success_rate" | tail -1
  done
done
echo "DONE-SLIP16-STEPS"
