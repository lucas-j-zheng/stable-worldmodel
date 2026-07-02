#!/bin/bash
#SBATCH -J ldm-slip-cache
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
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

# P1 Route 1 stage 2 (chained after slip_dose_screen via --dependency):
# GATE on the screen result, then cache slip latents through the EXISTING
# lewm_tworoom encoder — slip changes DYNAMICS only, not appearance, and the
# slip env uses the default 1-door geometry, so the encoder is in-distribution
# (no retrain). Then confirm the bimodality survives in LATENT space (the
# conditioning the dynamics model will actually use).

# Gate: hidden bimodal at S=8 must clear 0.3 (screen PASS), else abort — the
# standing screen-before-train rule, enforced in-job so the chain can be
# submitted before the screen finishes.
python - <<'EOF'
import json, sys
d = json.load(open('logs/multimodality_slip8_hidden.json'))
frac = d.get('residual_bimodality_fraction', 0.0)
print(f'[slip-cache] gate: slip8 hidden residual_bimodal = {frac:.3f}')
sys.exit(0 if frac > 0.3 else 1)
EOF

for S in 0 4 8; do
  echo "[slip-cache] caching tworoom_slip${S} latents ..."
  python scripts/data/cache_latents.py \
    lewm_checkpoint=lewm_tworoom/weights_epoch_300.pt \
    source=tworoom_slip${S}.lance \
    out_name=tworoom_slip${S}_latent.lance \
    device=cuda batch_size=256

  echo "[slip-cache] LATENT dynamics screen slip_scale=$S ..."
  python scripts/data/multimodality_diagnostic.py \
    --dataset tworoom_slip${S}_latent.lance --tag slip${S}_latdyn \
    --mode dynamics --target-col latent --cond-cols latent action \
    --k 96 --n-anchors 4000
done

echo "[slip-cache] SUMMARY (latent-space):"
python - <<'EOF'
import json, glob
for f in sorted(glob.glob('logs/multimodality_slip*_latdyn.json')):
    d = json.load(open(f))
    print(f"{d.get('tag', f):>14}: residual_bimodal="
          f"{d.get('residual_bimodality_fraction', float('nan')):.3f}")
EOF
echo "DONE-SLIP-CACHE"
