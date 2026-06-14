#!/bin/bash
#SBATCH -J ldm-ptype
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH --array=0-2
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err

set -euo pipefail

cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate

export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
export TWOROOM_DATA=${TWOROOM_DATA:-tworoom_expert.lance}
export TWOROOM_LATENT_DATA=${TWOROOM_LATENT_DATA:-tworoom_latent_e300.lance}
export LEWM_CKPT=${LEWM_CKPT:-lewm_tworoom/weights_epoch_300.pt}

mkdir -p logs "$STABLEWM_HOME/datasets" "$STABLEWM_HOME/checkpoints"

MAX_EPOCHS=${MAX_EPOCHS:-300}
NUM_DIFFUSION_SAMPLES=${NUM_DIFFUSION_SAMPLES:-8}

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) PTYPE=v  ;;
  1) PTYPE=x0 ;;
  2) PTYPE=eps ;;
  *) echo "Unknown task id" >&2; exit 2 ;;
esac

OUTPUT_MODEL_NAME="latent_diffusion_tworoom_${PTYPE}300"
DIFF_CKPT="${OUTPUT_MODEL_NAME}/weights_epoch_${MAX_EPOCHS}.pt"

echo "[ldm-ptype] task=${SLURM_ARRAY_TASK_ID} prediction_type=${PTYPE} -> ${OUTPUT_MODEL_NAME}"

python scripts/train/latent_diffusion.py \
  data=tworoom_latent \
  data.dataset.name="$TWOROOM_LATENT_DATA" \
  model.lewm_checkpoint="$LEWM_CKPT" \
  model.prediction_type="$PTYPE" \
  model.num_inference_steps=20 \
  output_model_name="$OUTPUT_MODEL_NAME" \
  trainer.max_epochs="$MAX_EPOCHS" \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  optimizer.lr=3e-4 \
  loader.batch_size=128 \
  num_workers=6 \
  loader.num_workers=6

python scripts/train/eval_latent_diffusion.py \
  data=tworoom \
  data.dataset.name="$TWOROOM_DATA" \
  diffusion_checkpoint="$DIFF_CKPT" \
  num_diffusion_samples="$NUM_DIFFUSION_SAMPLES"

echo "[ldm-ptype] === stage0 summary for ${PTYPE} ==="
cat "$STABLEWM_HOME/checkpoints/$OUTPUT_MODEL_NAME/latent_diffusion_stage0.json"
