#!/bin/bash
#SBATCH -J ldm-train-mm
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate

export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs "$STABLEWM_HOME/checkpoints"

# Train a v-parametrized diffusion DYNAMICS model on the multimodal-TwoRoom
# latents (random-committed-door demos). NB: this tests whether POLICY
# multimodality helps a DYNAMICS model -- expected NOT to, since TwoRoom dynamics
# are deterministic given the action. The closed-loop diffusion-vs-deterministic
# comparison (separate job) is the actual read.
MAX_EPOCHS=${MAX_EPOCHS:-250}
OUT=latent_diffusion_tworoom_mm_v

echo "[train-mm] $OUT on tworoom_mm0_latent.lance ($MAX_EPOCHS epochs)"
python scripts/train/latent_diffusion.py \
  data=tworoom_latent \
  data.dataset.name=tworoom_mm0_latent.lance \
  model.lewm_checkpoint=lewm_tworoom/weights_epoch_300.pt \
  model.prediction_type=v \
  model.num_inference_steps=20 \
  output_model_name="$OUT" \
  trainer.max_epochs="$MAX_EPOCHS" \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  optimizer.lr=3e-4 \
  loader.batch_size=128 \
  num_workers=6 \
  loader.num_workers=6

echo "[train-mm] done -> $OUT/weights_epoch_${MAX_EPOCHS}.pt"
