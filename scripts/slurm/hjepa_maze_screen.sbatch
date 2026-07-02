#!/bin/bash
#SBATCH -J hjepa-maze
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# H-JEPA E7: SECOND DOMAIN -- point-maze random-walk K-step screen.
# Runs regardless of E6's verdict: if E6 kills TwoRoom this is the escalation
# path (review E ladder); if E6 greenlights, the paper still needs a second
# domain (PushT died in R4).
#
# Random-walk maze = K-step occupancy: branches at junctions, no goal confound
# (no goal-directed policy at all), no demonstrator-preference confound.
#
# PRE-REGISTERED (HJEPA_REASONING.md E7):
#   1. full: composed determinism (det_R2 high, bimodal low) at all K.
#   2. none: bimodal_frac RISES with K as windows reach junction scale; at
#      small K (1-2) low (free-space diffusion), at K 8-16 substantially
#      higher (>= 0.2) IF junction spacing is within random-walk reach.
#   3. gap_ratio small in branched cells (walls make hard-separated modes).
#   4. Risk: random walk too slow to reach junctions within K=16 -> spread
#      stays local/unimodal; remedy = longer K or action-repeat, not a kill.

set -uo pipefail

cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate

export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

DS=pointmaze_medium_rand_T64.lance

# 1. collect (state-only, random policy, no rendering) — skip if already there
if [ ! -d "$STABLEWM_HOME/datasets/${DS}" ]; then
  python scripts/data/collect_maze_screen.py \
    --episodes 3000 --maze-type medium --steps 64 --num-envs 8 --out $DS
else
  echo "[hjepa-maze] dataset $DS exists, skipping collection"
fi

# 2. probe columns
python - << PY
import stable_worldmodel as swm
ds = swm.data.load_dataset('$DS')
print('[cols]', sorted(ds.column_names))
PY

# 3. K x coarse screen (upgraded instrument: GMM2 + gap_ratio come free)
for K in 1 2 4 8 16; do
  for COARSE in full sum none; do
    echo "[hjepa-maze] K=$K coarse=$COARSE ..."
    python scripts/data/multimodality_diagnostic.py \
      --dataset $DS --tag "hjepa_maze_k${K}_${COARSE}" \
      --mode dynamics_k --horizon $K --coarse-action $COARSE \
      --target-col observation --cond-cols observation \
      --k 96 --n-anchors 4000 \
      || echo "[hjepa-maze] SKIP K=$K $COARSE"
  done
done

echo "[hjepa-maze] DONE. JSONs: logs/multimodality_hjepa_maze_*.json"
