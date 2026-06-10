#!/bin/bash
#SBATCH -J ldm-smoke
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate

export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
export TWOROOM_DATA=${TWOROOM_DATA:-tworoom_expert.lance}
export LEWM_CKPT=${LEWM_CKPT:-lewm_tworoom_smoke/weights_epoch_1.pt}
export TWOROOM_LATENT_DATA=${TWOROOM_LATENT_DATA:-tworoom_latent_smoke.lance}
export DIFF_CKPT=${DIFF_CKPT:-latent_diffusion_tworoom_smoke/weights_epoch_1.pt}

mkdir -p logs "$STABLEWM_HOME/datasets" "$STABLEWM_HOME/checkpoints"

echo "[latent-dmpc-smoke] STABLEWM_HOME=$STABLEWM_HOME"
echo "[latent-dmpc-smoke] TWOROOM_DATA=$TWOROOM_DATA"
echo "[latent-dmpc-smoke] LEWM_CKPT=$LEWM_CKPT"
echo "[latent-dmpc-smoke] TWOROOM_LATENT_DATA=$TWOROOM_LATENT_DATA"
echo "[latent-dmpc-smoke] DIFF_CKPT=$DIFF_CKPT"

python - <<'PY'
import json
import os
from pathlib import Path

p = (
    Path(os.environ['STABLEWM_HOME'])
    / 'checkpoints'
    / 'lewm_tworoom_smoke'
    / 'config.json'
)
if not p.exists():
    raise FileNotFoundError(f'Missing LeWM config: {p}')

cfg = json.loads(p.read_text())
if '_target_' not in cfg and 'model' in cfg:
    p.with_name('config.full.json').write_text(json.dumps(cfg, indent=2))
    p.write_text(json.dumps(cfg['model'], indent=2))
    cfg = cfg['model']

print(f'[latent-dmpc-smoke] LeWM config target: {cfg["_target_"]}')
PY

python scripts/data/cache_latents.py \
  lewm_checkpoint="$LEWM_CKPT" \
  source="$TWOROOM_DATA" \
  out_name="$TWOROOM_LATENT_DATA" \
  device=cuda \
  batch_size=256

python - <<'PY'
import os
import stable_worldmodel as swm

ds = swm.data.load_dataset(
    os.environ['TWOROOM_LATENT_DATA'],
    num_steps=11,
    frameskip=5,
    keys_to_load=['latent', 'action'],
)
print(f'[latent-dmpc-smoke] latent dataset length: {len(ds)}')
print(f'[latent-dmpc-smoke] latent dataset columns: {ds.column_names}')
PY

python scripts/train/latent_diffusion.py \
  data=tworoom_latent \
  data.dataset.name="$TWOROOM_LATENT_DATA" \
  model.lewm_checkpoint="$LEWM_CKPT" \
  output_model_name=latent_diffusion_tworoom_smoke \
  trainer.max_epochs=1 \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  loader.batch_size=32 \
  num_workers=4 \
  loader.num_workers=4

python scripts/train/eval_latent_diffusion.py \
  data=tworoom \
  data.dataset.name="$TWOROOM_DATA" \
  diffusion_checkpoint="$DIFF_CKPT" \
  num_diffusion_samples=4

cat "$STABLEWM_HOME/checkpoints/latent_diffusion_tworoom_smoke/latent_diffusion_stage0.json"
