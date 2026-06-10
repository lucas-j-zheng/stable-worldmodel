#!/bin/bash
#SBATCH -J ldm-stage0
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH --array=0-5
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err

set -euo pipefail

cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate

export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
export TWOROOM_DATA=${TWOROOM_DATA:-tworoom_expert.lance}
export LEWM_CKPT=${LEWM_CKPT:-lewm_tworoom/weights_epoch_100.pt}
export TWOROOM_LATENT_DATA=${TWOROOM_LATENT_DATA:-tworoom_latent.lance}

mkdir -p logs "$STABLEWM_HOME/datasets" "$STABLEWM_HOME/checkpoints"

MAX_EPOCHS=${MAX_EPOCHS:-25}
NUM_DIFFUSION_SAMPLES=${NUM_DIFFUSION_SAMPLES:-8}

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0)
    RUN_SUFFIX=base
    EXTRA=(model.num_diffusion_steps=100 model.num_inference_steps=10 model.eta=0.0 model.denoiser.depth=6 optimizer.lr=1e-4)
    ;;
  1)
    RUN_SUFFIX=infer20
    EXTRA=(model.num_diffusion_steps=100 model.num_inference_steps=20 model.eta=0.0 model.denoiser.depth=6 optimizer.lr=1e-4)
    ;;
  2)
    RUN_SUFFIX=eta05
    EXTRA=(model.num_diffusion_steps=100 model.num_inference_steps=10 model.eta=0.5 model.denoiser.depth=6 optimizer.lr=1e-4)
    ;;
  3)
    RUN_SUFFIX=depth4
    EXTRA=(model.num_diffusion_steps=100 model.num_inference_steps=10 model.eta=0.0 model.denoiser.depth=4 optimizer.lr=1e-4)
    ;;
  4)
    RUN_SUFFIX=depth8
    EXTRA=(model.num_diffusion_steps=100 model.num_inference_steps=10 model.eta=0.0 model.denoiser.depth=8 optimizer.lr=1e-4)
    ;;
  5)
    RUN_SUFFIX=lr3e4
    EXTRA=(model.num_diffusion_steps=100 model.num_inference_steps=10 model.eta=0.0 model.denoiser.depth=6 optimizer.lr=3e-4)
    ;;
  *)
    echo "Unknown SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-unset}" >&2
    exit 2
    ;;
esac

OUTPUT_MODEL_NAME="latent_diffusion_tworoom_${RUN_SUFFIX}"
DIFF_CKPT="${OUTPUT_MODEL_NAME}/weights_epoch_${MAX_EPOCHS}.pt"

echo "[latent-dmpc-stage0] task=${SLURM_ARRAY_TASK_ID:-0}"
echo "[latent-dmpc-stage0] output=${OUTPUT_MODEL_NAME}"
echo "[latent-dmpc-stage0] LEWM_CKPT=${LEWM_CKPT}"
echo "[latent-dmpc-stage0] TWOROOM_DATA=${TWOROOM_DATA}"
echo "[latent-dmpc-stage0] TWOROOM_LATENT_DATA=${TWOROOM_LATENT_DATA}"
printf '[latent-dmpc-stage0] overrides:'
printf ' %q' "${EXTRA[@]}"
printf '\n'

python scripts/train/latent_diffusion.py \
  data=tworoom_latent \
  data.dataset.name="$TWOROOM_LATENT_DATA" \
  model.lewm_checkpoint="$LEWM_CKPT" \
  output_model_name="$OUTPUT_MODEL_NAME" \
  trainer.max_epochs="$MAX_EPOCHS" \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  loader.batch_size=128 \
  num_workers=6 \
  loader.num_workers=6 \
  "${EXTRA[@]}"

python scripts/train/eval_latent_diffusion.py \
  data=tworoom \
  data.dataset.name="$TWOROOM_DATA" \
  diffusion_checkpoint="$DIFF_CKPT" \
  num_diffusion_samples="$NUM_DIFFUSION_SAMPLES"

cat "$STABLEWM_HOME/checkpoints/$OUTPUT_MODEL_NAME/latent_diffusion_stage0.json"
