#!/bin/bash
#SBATCH -J lewm-tworoom
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate

export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
export TWOROOM_DATA=${TWOROOM_DATA:-tworoom_expert.lance}

mkdir -p logs "$STABLEWM_HOME/datasets" "$STABLEWM_HOME/checkpoints"

echo "[train-lewm-tworoom] STABLEWM_HOME=$STABLEWM_HOME"
echo "[train-lewm-tworoom] TWOROOM_DATA=$TWOROOM_DATA"

python scripts/train/lewm.py \
  data=tworoom \
  data.dataset.name="$TWOROOM_DATA" \
  output_model_name=lewm_tworoom \
  trainer.max_epochs=100 \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  loader.batch_size=128 \
  num_workers=6 \
  loader.num_workers=6
