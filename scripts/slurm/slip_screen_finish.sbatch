#!/bin/bash
#SBATCH -J ldm-slip-finish
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate
export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

# Finish the slip dose screen after 3612481 timed out mid-slip8-collection.
# slip0/2/4 landed (hidden bimodal 0.114 / 0.975 / 0.987 — gate PASSED); the
# slip8 dataset is truncated (392M vs 1.1G) -> recollect from scratch, screen.
# NB the "observed" control is OFF-BY-ONE (slip_state on row t is the slip
# that produced state_t, the screened transition needs row t+1's slip) — the
# real control is the slip0-vs-slipN dose, which is unambiguous.

rm -rf "$STABLEWM_HOME/datasets/tworoom_slip8.lance"
echo "[slip-finish] recollecting tworoom_slip8.lance ..."
python scripts/data/collect_tworooms_slip.py \
  cache_dir="$STABLEWM_HOME" num_traj=4000 \
  world.num_envs=10 world.max_episode_steps=100 \
  +slip_scale=8 +action_noise=2.0 +out_name=tworoom_slip8.lance

echo "[slip-finish] screen HIDDEN slip8 ..."
python scripts/data/multimodality_diagnostic.py \
  --dataset tworoom_slip8.lance --tag slip8_hidden \
  --mode dynamics --target-col state --cond-cols state action \
  --k 96 --n-anchors 4000

python - <<'EOF'
import json
for t in ['slip0', 'slip2', 'slip4', 'slip8']:
    d = json.load(open(f'logs/multimodality_{t}_hidden.json'))
    print(f"[slip-finish] {t}: residual_bimodal="
          f"{d.get('residual_bimodality_fraction', float('nan')):.3f}")
EOF
echo "DONE-SLIP-FINISH"
