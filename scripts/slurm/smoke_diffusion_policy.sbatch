#!/bin/bash
#SBATCH -J ldm-dp-smoke
#SBATCH -p gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 00:40:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate
export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

# SMOKE: validate the diffusion-policy + MSE-baseline training paths end-to-end
# (shapes, loss, checkpoint save) on existing tworoom latents. 3 epochs each.
echo "[smoke] DiffusionPolicy ..."
python scripts/train/train_diffusion_policy.py \
  data=tworoom_latent data.dataset.name=tworoom_latent_e300.lance \
  model.lewm_checkpoint=lewm_tworoom/weights_epoch_300.pt \
  output_model_name=dp_smoke train_module=model.denoiser \
  trainer.max_epochs=3 trainer.accelerator=gpu trainer.devices=1 \
  loader.batch_size=64 num_workers=4 loader.num_workers=4 checkpoint_interval=3

echo "[smoke] MLP baseline ..."
python scripts/train/train_diffusion_policy.py \
  data=tworoom_latent data.dataset.name=tworoom_latent_e300.lance \
  model._target_=stable_worldmodel.wm.diffusion_policy.MLPPolicy \
  ~model.denoiser ~model.num_diffusion_steps ~model.schedule \
  ~model.num_inference_steps ~model.eta ~model.prediction_type ~model.clip_sample \
  model.lewm_checkpoint=lewm_tworoom/weights_epoch_300.pt \
  output_model_name=mlp_smoke train_module=model.net \
  trainer.max_epochs=3 trainer.accelerator=gpu trainer.devices=1 \
  loader.batch_size=64 num_workers=4 loader.num_workers=4 checkpoint_interval=3

echo "[smoke] OK -- both training paths ran."
