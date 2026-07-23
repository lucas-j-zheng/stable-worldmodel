#!/bin/bash
#SBATCH -J swm-restore-pusht
#SBATCH -p gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Scratch-purge recovery (diagnosis 2026-07-23, discovered 2026-07-24): the
# purge emptied lewm_pusht/ (frozen encoder) and deleted pusht_expert_train.h5
# (base-pusht pixels). Both are restorable from HF (quentinll/lewm-pusht).
# Also back up the surviving pusht-program checkpoints to home (purge-free).
set -uo pipefail
cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
mkdir -p logs

echo "[restore] 1/3 refetch base-pusht dataset (HF .h5.zst -> .h5) ..."
if [ ! -e "$STABLEWM_HOME/datasets/pusht_expert_train.h5" ]; then
  python scripts/data/fetch_lewm_dataset.py \
    --repo quentinll/lewm-pusht --out-dir "$STABLEWM_HOME/datasets"
else
  echo "[restore] dataset already present, skipping"
fi
ls -la "$STABLEWM_HOME/datasets/"pusht_expert* || { echo "DATASET RESTORE FAILED"; exit 1; }

echo "[restore] 2/3 rebuild lewm_pusht from quentinll/lewm-pusht ..."
if [ ! -e "$STABLEWM_HOME/checkpoints/lewm_pusht/weights.pt" ]; then
  python scripts/data/convert_lewm_checkpoint.py \
    --src-repo quentinll/lewm-pusht --out lewm_pusht
else
  echo "[restore] lewm_pusht already present, skipping"
fi
ls -la "$STABLEWM_HOME/checkpoints/lewm_pusht/" || { echo "ENCODER RESTORE FAILED"; exit 1; }

echo "[restore] sanity: encoder loads + encodes ..."
python - <<'EOF'
import torch
from stable_worldmodel.wm.utils import load_pretrained
m = load_pretrained('lewm_pusht/weights.pt').cuda().eval()
x = torch.randn(1, 2, 3, 224, 224).cuda()
with torch.no_grad():
    emb = m.encode({'pixels': x})['emb']
print('[restore] encode OK:', tuple(emb.shape))
EOF

echo "[restore] 3/3 backup pusht-program checkpoints to ~/swm_backup ..."
mkdir -p ~/swm_backup/checkpoints
for d in lewm_pusht dyn_v_pusht_ps1 dyn_v_pusht_ps2 dyn_v_pusht_ps3 \
         dyn_x0_pusht_ps1 dyn_x0_pusht_ps2 dyn_x0_pusht_ps3 \
         dyn_tmse_pusht_ps1 dyn_tmse_pusht_ps2 dyn_tmse_pusht_ps3 \
         e2c_diff_cslip05_ps1 e2c_diff_cslip05_ps2 \
         e2c_diff_sw0_cslip05_ps1 e2c_diff_sw0_cslip05_ps2 e2c_diff_sw0_cslip05_ps3 \
         e2c_tmse_cslip05_ps1 e2c_tmse_cslip05_ps2 e2c_tmse_cslip05_ps3; do
  src="$STABLEWM_HOME/checkpoints/$d"
  [ -d "$src" ] && rsync -a --exclude '*.mp4' "$src" ~/swm_backup/checkpoints/
done
du -sh ~/swm_backup/checkpoints
echo "DONE-RESTORE"
