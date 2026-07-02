#!/bin/bash
#SBATCH -J ldm-slip-dmpc
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

# P1 Route 1 VERDICT (chained after slip_dynamics_cell via --dependency):
# closed-loop D-MPC (CEM) on the slip env — diffusion dynamics vs the
# same-backbone TransformerMSEDynamics, 3 seeds x 2 eval seeds x n=50.
#   TAG=slip8 SLIP=8 -> the multimodal-dynamics cell (thesis predicts diffusion
#     wins: the deterministic model averages the two slip branches).
#   TAG=slip0 SLIP=0 -> unimodal control (thesis predicts a tie).
# Positions npz written per eval (failure geometry, same probe as P0a).
TAG=${TAG:?set TAG=slip8 or slip0}
SLIP=${SLIP:?set SLIP=8 or 0}
EP=${EP:-250}

for s in 1 2 3; do
  for kind in diff tmse; do
    name="dyn_${kind}_${TAG}_ps${s}"
    for es in 42 123; do
      echo "[slip-dmpc] ==== EVAL $name (offset 25, n=50, eval seed $es) ===="
      python scripts/plan/eval_wm.py --config-name tworoom_slip_diffusion \
        policy=${name}/weights_epoch_$EP.pt cache_dir="$LOCAL_DATASET_DIR" \
        seed=$es world.slip_scale=$SLIP eval.dataset_name=tworoom_${TAG}.lance \
        2>&1 | grep -iE "success_rate|positions saved" | tail -2
    done
  done
done
echo "DONE-SLIP-DMPC-$TAG"
