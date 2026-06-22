#!/bin/bash
#SBATCH -J ldm-lewm-mm
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate
export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs "$STABLEWM_HOME/checkpoints"

# Retrain the LeWM encoder on the 2-DOOR multimodal data so the eval scene is
# in-distribution (the earlier mm-dynamics eval scored 0%/0% because the 1-door
# encoder was OOD on 2-door scenes). Required before training/eval the policies.
MAX_EPOCHS=${MAX_EPOCHS:-300}
echo "[lewm-mm] training encoder on tworoom_mm0.lance ($MAX_EPOCHS epochs)"
python scripts/train/lewm.py \
  data=tworoom \
  data.dataset.name=tworoom_mm0.lance \
  output_model_name=lewm_tworoom_mm \
  trainer.max_epochs="$MAX_EPOCHS" \
  warmup_pct=0.05 \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  loader.batch_size=128 \
  num_workers=6 \
  loader.num_workers=6

echo "[lewm-mm] done -> lewm_tworoom_mm/weights_epoch_${MAX_EPOCHS}.pt"
