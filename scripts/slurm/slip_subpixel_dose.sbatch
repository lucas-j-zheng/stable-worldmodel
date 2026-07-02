#!/bin/bash
#SBATCH -J ldm-slip-subpix
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 02:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate
export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

# Graded slip dose: the integer sweep saturates instantly (0.114 at S=0 ->
# 0.975 at S=2), so the paper's dose curve needs sub-pixel separations to
# resolve the transition. Locates the detection threshold of the screen (where
# 2*S crosses the kNN/action-noise floor) and gives a graded x-axis.
for S in 0.25 0.5 1; do
  DS=tworoom_slip_s${S}.lance
  echo "[subpix] collecting $DS (slip_scale=$S) ..."
  python scripts/data/collect_tworooms_slip.py \
    cache_dir="$STABLEWM_HOME" num_traj=4000 \
    world.num_envs=10 world.max_episode_steps=100 \
    +slip_scale=$S +action_noise=2.0 +out_name=$DS

  echo "[subpix] screen HIDDEN slip_scale=$S ..."
  python scripts/data/multimodality_diagnostic.py \
    --dataset $DS --tag slip_s${S}_hidden \
    --mode dynamics --target-col state --cond-cols state action \
    --k 96 --n-anchors 4000
done

python - <<'EOF'
import json, glob
for f in sorted(glob.glob('logs/multimodality_slip*_hidden.json')):
    d = json.load(open(f))
    print(f"[subpix] {d.get('tag', f):>16}: residual_bimodal="
          f"{d.get('residual_bimodality_fraction', float('nan')):.3f}")
EOF
echo "DONE-SLIP-SUBPIX"
