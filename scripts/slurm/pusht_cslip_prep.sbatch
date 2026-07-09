#!/bin/bash
#SBATCH -J ldm-pusht-cslip-prep
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

# ARC 4 stage 2: cache contact-slip PushT latents through the frozen lewm_pusht
# encoder (slip changes DYNAMICS not appearance -> encoder in-distribution) and
# confirm the multimodality survives in LATENT space (what the dynamics model
# actually conditions on). Raw-state screen already showed strong bimodality
# (~0.98) and a det_R2 dose-response (0.42@S0.5 -> 0.20@S1.0 = the coin injects
# aleatoric stochasticity); this checks it persists post-encoder.
ENC=lewm_pusht/weights.pt

for TAG in cslip05 cslip10; do
  DS=pusht_${TAG}.lance
  LAT=pusht_${TAG}_latent.lance
  echo "[cslip-prep] caching $DS -> $LAT ..."
  python scripts/data/cache_latents.py \
    lewm_checkpoint=$ENC source=$DS out_name=$LAT \
    device=cuda batch_size=256

  echo "[cslip-prep] LATENT dynamics screen $TAG ..."
  python scripts/data/multimodality_diagnostic.py \
    --dataset $LAT --tag pusht_${TAG}_latdyn \
    --mode dynamics --target-col latent --cond-cols latent action \
    --k 96 --n-anchors 4000
done

echo "[cslip-prep] SUMMARY (latent-space):"
python - <<'EOF'
import json, glob
for f in sorted(glob.glob('logs/multimodality_pusht_cslip*_latdyn.json')):
    d = json.load(open(f))
    print(f"{d.get('tag', f):>24}: residual_bimodal="
          f"{d.get('residual_bimodality_fraction', float('nan')):.3f}")
EOF
echo "DONE-PUSHT-CSLIP-PREP"
