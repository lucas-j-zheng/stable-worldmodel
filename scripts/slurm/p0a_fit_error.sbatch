#!/bin/bash
#SBATCH -J ldm-p0a-fiterr
#SBATCH -p gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate
export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

# P0a smoothing test (mechanism candidate a) on the EXISTING endpoint
# checkpoints: per-sample fit error vs the demo chunk, normalized action space,
# train+val. Prediction if training-time mode-averaging is the mechanism:
# tmse mm05 error >> tmse p10 error, gap larger than dp's.
# Seeds match multiseed_cell training (cfg.seed controlled the split there).

run_fit () {  # $1=kind $2=tag(mm05|p10) $3=data $4=seed
  local ck="${1}_${2}_s${4}/weights_epoch_250.pt"
  echo "[fiterr] $ck on $3 ..."
  python scripts/data/policy_fit_error.py data=tworoom_latent \
    data.dataset.name=$3 seed=$4 \
    +policy_ckpt=$ck +tag=${1}_${2}_s${4} \
    loader.batch_size=128 num_workers=6 loader.num_workers=6
}

for s in 1 2 3; do
  for kind in dp tmse; do
    run_fit $kind mm05 tworoom_mm_latent2.lance $s
    run_fit $kind p10  tworoom_p10_latent.lance $s
  done
done

echo "[fiterr] SUMMARY (train/val mean per-sample normalized MSE):"
python - <<'EOF'
import json, glob
for f in sorted(glob.glob('logs/fit_error_*_s*.json')):
    d = json.load(open(f))
    print(f"{d['tag']:>14}  train={d['train']['mean']:.4f}  val={d['val']['mean']:.4f}")
EOF
echo "DONE-P0A-FITERR"
