#!/bin/bash
#SBATCH -J ldm-slip-screen
#SBATCH -p gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# P1 Route 1 GATE: does the intrinsic per-step SLIP make p(next|state,action)
# genuinely and dose-dependently multimodal? Encoder-free, cheap — the standing
# rule (screen BEFORE any diffusion training) applied to the slip env.
#   PASS = hidden residual_bimodal ~0 at S=0, HIGH and rising with S in {2,4,8};
#          observed (slip_state in cond) stays ~0 at every S (control: the coin
#          explains the split). -> greenlight encoder+dynamics pipeline on slip.
#   FAIL = flat/low even at S=8 -> separation drowned by action_noise/kNN floor;
#          raise S or lower action_noise before any GPU spend.

set -euo pipefail
cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate
export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

for S in 0 2 4 8; do
  DS=tworoom_slip${S}.lance
  echo "[slip] collecting $DS (slip_scale=$S) ..."
  python scripts/data/collect_tworooms_slip.py \
    cache_dir="$STABLEWM_HOME" num_traj=4000 \
    world.num_envs=10 world.max_episode_steps=100 \
    +slip_scale=$S +action_noise=2.0 +out_name=$DS

  echo "[slip] screen HIDDEN slip_scale=$S ..."
  python scripts/data/multimodality_diagnostic.py \
    --dataset $DS --tag slip${S}_hidden \
    --mode dynamics --target-col state --cond-cols state action \
    --k 96 --n-anchors 4000

  echo "[slip] screen OBSERVED slip_scale=$S (control) ..."
  python scripts/data/multimodality_diagnostic.py \
    --dataset $DS --tag slip${S}_observed \
    --mode dynamics --target-col state --cond-cols state slip_state action \
    --k 96 --n-anchors 4000
done

echo "[slip] SUMMARY:"
python - <<'EOF'
import json, glob
for f in sorted(glob.glob('logs/multimodality_slip*_*.json')):
    d = json.load(open(f))
    tag = d.get('tag', f)
    print(f"{tag:>16}: det_R2={d.get('det_R2', d.get('det_r2', float('nan'))):.3f} "
          f"residual_bimodal={d.get('residual_bimodality_fraction', float('nan')):.3f}")
EOF
echo "DONE-SLIP-SCREEN"
