#!/bin/bash
#SBATCH -J ldm-pusht-cslip10-lat
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

# ARC 4 robustness check: does the encoder's multimodality COLLAPSE hold at the
# STRONGER dose? cslip05 already showed raw 0.98 -> latent 0.029. Cache ONLY
# cslip10 (one dataset — the 2-dataset prep timed out at 2h) and screen its
# latent dynamics. If cslip10 latent is also ~0 despite raw 0.98 & det_R2 0.20,
# the frozen encoder is a dose-independent multimodality bottleneck (capstone).
ENC=lewm_pusht/weights.pt
TAG=cslip10
DS=pusht_${TAG}.lance
LAT=pusht_${TAG}_latent.lance

if [ ! -d "$STABLEWM_HOME/datasets/$LAT" ]; then
  echo "[cslip10-lat] caching $DS -> $LAT ..."
  python scripts/data/cache_latents.py \
    lewm_checkpoint=$ENC source=$DS out_name=$LAT device=cuda batch_size=256
else
  echo "[cslip10-lat] $LAT already cached, skipping cache."
fi

echo "[cslip10-lat] LATENT dynamics screen $TAG ..."
python scripts/data/multimodality_diagnostic.py \
  --dataset $LAT --tag pusht_${TAG}_latdyn \
  --mode dynamics --target-col latent --cond-cols latent action \
  --k 96 --n-anchors 4000

python - <<'EOF'
import json
d = json.load(open('logs/multimodality_pusht_cslip10_latdyn.json'))
print(f"[cslip10-lat] cslip10 LATENT residual_bimodal = "
      f"{d.get('residual_bimodality_fraction', float('nan')):.3f} "
      f"(raw was 0.978; cslip05 latent was 0.029)")
EOF
echo "DONE-PUSHT-CSLIP10-LATSCREEN"
