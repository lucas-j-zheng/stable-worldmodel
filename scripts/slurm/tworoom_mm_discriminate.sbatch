#!/bin/bash
#SBATCH -J ldm-tworoom-disc
#SBATCH -p gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate
export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

# Confirm available columns (does goal_state/target exist?).
echo "[disc] columns:"
python -c "
import stable_worldmodel as swm
ds = swm.data.load_dataset('tworoom_mm0_latent.lance')
print('  cols:', sorted(ds.column_names))
import numpy as np
ep = ds.load_episode(0)
for k in ds.column_names:
    try: print('  ', k, np.asarray(ep[k]).shape)
    except Exception: pass
"

# THE DISCRIMINATOR: condition on agent+target together. greedy door is a
# deterministic function of (agent,target,doors) -> residual ~0; the multimodal
# door is random per episode -> residual stays HIGH. This isolates the injected
# door-choice multimodality that weak (agent-only / latent) conditioning missed.
for d in tworoom_mm0 tworoom_greedy0; do
  echo "[disc] $d: policy cond=[state,goal_state] ..."
  python scripts/data/multimodality_diagnostic.py \
    --dataset ${d}_latent.lance --tag ${d}_disc_policy \
    --mode policy --cond-cols state goal_state || \
  echo "[disc] (goal_state likely absent -- see columns above)"
done

echo "[disc] done."
