#!/bin/bash
#SBATCH -J ldm-pusht-slip-screen
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate
export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

# ARC 4 GATE (screen BEFORE any diffusion training — the standing rule): does the
# intrinsic per-step CONTACT-SLIP make p(next|state,action) genuinely multimodal in
# PushT? Encoder-free, raw state.
#   PASS = hidden residual_bimodal ~0 at S=0 sense, HIGH at S in {0.5,1.0}; the
#          OBSERVED screen (contact_slip in cond) drops back toward ~0 (control:
#          the coin explains the split) -> greenlight the dynamics train+eval.
#   FAIL = flat/low even at S=1.0 -> perturbation too small or washed out by
#          contact geometry; raise S before any GPU training spend.
# Unlike TwoRoom slip, these modes are IRREVERSIBLE (contact-mediated) — the whole
# point: the one regime where planning to the conditional mean should fail.

for S in 0.5 1.0; do
  tag=$(echo "$S" | tr -d '.')
  DS=pusht_cslip${tag}.lance
  echo "[pusht-slip] collecting $DS (contact_slip_scale=$S) ..."
  python scripts/data/collect_pusht_slip.py \
    cache_dir="$STABLEWM_HOME" num_traj=1500 \
    world.num_envs=10 world.max_episode_steps=120 \
    +contact_slip_scale=$S +out_name=$DS

  echo "[pusht-slip] screen HIDDEN S=$S ..."
  python scripts/data/multimodality_diagnostic.py \
    --dataset $DS --tag pcslip${tag}_hidden \
    --mode dynamics --target-col state --cond-cols state action \
    --k 96 --n-anchors 4000

  echo "[pusht-slip] screen OBSERVED S=$S (control) ..."
  python scripts/data/multimodality_diagnostic.py \
    --dataset $DS --tag pcslip${tag}_observed \
    --mode dynamics --target-col state --cond-cols state contact_slip action \
    --k 96 --n-anchors 4000
done

echo "[pusht-slip] SUMMARY:"
python - <<'EOF'
import json, glob
for f in sorted(glob.glob('logs/multimodality_pcslip*_*.json')):
    d = json.load(open(f))
    tag = d.get('tag', f)
    print(f"{tag:>20}: det_R2={d.get('det_R2', d.get('det_r2', float('nan'))):.3f} "
          f"residual_bimodal={d.get('residual_bimodality_fraction', float('nan')):.3f}")
EOF
echo "DONE-PUSHT-SLIP-SCREEN"
