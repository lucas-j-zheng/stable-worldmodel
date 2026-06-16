#!/bin/bash
#SBATCH -J ldm-lerobot-convert
#SBATCH -p gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel

# lerobot/pusht stores `pixels` as VIDEO, decoded by torchcodec, which needs
# FFmpeg shared libs (libavutil/...). The bare node doesn't expose them ->
# load Oscar's ffmpeg module so torchcodec finds a matching libav* (ffmpeg 7.1
# -> libavutil.so.59; torchcodec probes ffmpeg 4-7 and picks the match).
module load ffmpeg/7.1-7dmq 2>/dev/null || module load ffmpeg 2>/dev/null || true
echo "[lerobot] ffmpeg libs: $(ls ${EBROOTFFMPEG:-/nonexistent}/lib/libavutil.so* 2>/dev/null | tr '\n' ' ')"

export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
export HF_HOME=${HF_HOME:-$STABLEWM_HOME/hf_cache}     # keep HF cache off home quota
export UV=${UV:-$HOME/.local/bin/uv}
export MUJOCO_GL=egl
mkdir -p logs "$STABLEWM_HOME/datasets" "$HF_HOME"

# --- 1. Build an ISOLATED Python 3.12 venv for LeRobot data ingestion only.
# Kept separate from the working 3.10 .venv: the two communicate purely through
# the on-disk .lance, so the 3.10 training/eval pipeline is never touched.
# Minimal stack: base swm + lerobot + transformers (the latter only because
# `import stable_worldmodel` -> wm -> prejepa imports transformers at module
# load). Skip the full `train` extra (stable-pretraining/hydra/wandb) -> avoids
# their 3.12 resolution risk; convert.py is argparse-only and doesn't need them.
if [ ! -x .venv312/bin/python ]; then
  echo "[lerobot] building .venv312 (python 3.12) ..."
  "$UV" venv --python 3.12 .venv312
fi
echo "[lerobot] installing base + lerobot + transformers into .venv312 ..."
VIRTUAL_ENV="$PWD/.venv312" "$UV" pip install -e '.[lerobot]' transformers

PY=.venv312/bin/python
echo "[lerobot] python: $($PY --version)"

# --- 2. Sanity: imports + dataset column/shape inspection (learn state dims).
echo "[lerobot] import + inspect ..."
$PY - <<'PYEOF'
import lerobot, transformers
print('[lerobot] lerobot', lerobot.__version__, '| transformers', transformers.__version__)
from stable_worldmodel.data import load_dataset
ds = load_dataset('lerobot://lerobot/pusht_image')
print('[lerobot] columns:', sorted(ds.column_names))
print('[lerobot] n_episodes:', len(ds.lengths), '| first 3 lengths:', ds.lengths[:3].tolist())
ep = ds.load_episode(0)
for k, v in ep.items():
    try:
        import numpy as np
        a = np.asarray(v)
        print(f'[lerobot]   {k}: shape={a.shape} dtype={a.dtype}')
    except Exception as e:
        print(f'[lerobot]   {k}: <{type(v).__name__}> ({e})')
PYEOF

# --- 3. Convert lerobot/pusht (206 human-teleop episodes) -> lance for the
# existing 3.10 pipeline (cache_latents -> multimodality diagnostic -> train).
OUT="$STABLEWM_HOME/datasets/pusht_human.lance"
echo "[lerobot] converting lerobot://lerobot/pusht_image -> $OUT"
$PY scripts/data/convert.py \
  --source lerobot://lerobot/pusht_image \
  --dest "$OUT" \
  --mode overwrite

echo "[lerobot] DONE. wrote $OUT"
