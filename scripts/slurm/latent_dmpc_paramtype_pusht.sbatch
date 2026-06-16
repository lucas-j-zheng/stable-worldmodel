#!/bin/bash
#SBATCH -J ldm-ptype-pusht
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 16:00:00
#SBATCH --array=0-1
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err

set -euo pipefail

cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate

export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
export PUSHT_DATA=${PUSHT_DATA:-pusht_expert_train.h5}
export PUSHT_LATENT_DATA=${PUSHT_LATENT_DATA:-pusht_latent.lance}
export LEWM_CKPT=${LEWM_CKPT:-lewm_pusht/weights.pt}

mkdir -p logs

# PushT is ~2M frames (~15.6k batches/epoch at bs=128); keep epochs small.
MAX_EPOCHS=${MAX_EPOCHS:-5}
NUM_DIFFUSION_SAMPLES=${NUM_DIFFUSION_SAMPLES:-8}

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) PTYPE=v  ;;
  1) PTYPE=x0 ;;
  2) PTYPE=eps ;;
  *) echo "Unknown task id" >&2; exit 2 ;;
esac

OUTPUT_MODEL_NAME="latent_diffusion_pusht_${PTYPE}"
DIFF_CKPT="${OUTPUT_MODEL_NAME}/weights_epoch_${MAX_EPOCHS}.pt"

echo "[ldm-ptype-pusht] task=${SLURM_ARRAY_TASK_ID} prediction_type=${PTYPE} -> ${OUTPUT_MODEL_NAME}"

python scripts/train/latent_diffusion.py \
  data=pusht_latent \
  data.dataset.name="$PUSHT_LATENT_DATA" \
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
  data=pusht \
  data.dataset.name="$PUSHT_DATA" \
  diffusion_checkpoint="$DIFF_CKPT" \
  num_diffusion_samples="$NUM_DIFFUSION_SAMPLES"

echo "[ldm-ptype-pusht] === stage0 summary for ${PTYPE} ==="
cat "$STABLEWM_HOME/checkpoints/$OUTPUT_MODEL_NAME/latent_diffusion_stage0.json"
