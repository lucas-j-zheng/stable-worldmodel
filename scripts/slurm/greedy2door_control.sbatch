#!/bin/bash
#SBATCH -J ldm-greedy2
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate
export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

# DECISIVE CONTROL: greedy CLOSEST-of-2-doors on the SAME 2-door env/encoder.
# This is the FAIR unimodal control -- natural efficient routes (unlike the
# pathological door_prob=1.0 "always far door"). Holds env/encoder/difficulty
# fixed vs the multimodal experiment; only the demos' multimodality differs.
#   thesis holds  -> DP ~= MLP here (no multimodality, no diffusion edge)
#   thesis refuted -> DP >> MLP here too (the 2-door task just favors diffusion)
# tworoom_greedy0.lance already exists (2-door geometry, greedy, noise-free);
# just re-cache it with the 2-DOOR encoder.
ENC=lewm_tworoom_mm/weights_epoch_297.pt
EP=${EP:-250}

echo "[greedy2] cache greedy-2door latents with 2-door encoder ..."
python scripts/data/cache_latents.py lewm_checkpoint=$ENC \
  source=tworoom_greedy0.lance out_name=tworoom_greedy2_latent.lance device=cuda batch_size=256

echo "[greedy2] train DiffusionPolicy ..."
python scripts/train/train_diffusion_policy.py \
  data=tworoom_latent data.dataset.name=tworoom_greedy2_latent.lance \
  model.lewm_checkpoint=$ENC output_model_name=dp_greedy2 train_module=model.denoiser \
  trainer.max_epochs=$EP trainer.accelerator=gpu trainer.devices=1 \
  loader.batch_size=128 num_workers=6 loader.num_workers=6 checkpoint_interval=$EP

echo "[greedy2] train MLP ..."
python scripts/train/train_diffusion_policy.py \
  data=tworoom_latent data.dataset.name=tworoom_greedy2_latent.lance \
  model._target_=stable_worldmodel.wm.diffusion_policy.MLPPolicy \
  ~model.denoiser ~model.num_diffusion_steps ~model.schedule \
  ~model.num_inference_steps ~model.eta ~model.prediction_type ~model.clip_sample \
  model.lewm_checkpoint=$ENC output_model_name=mlp_greedy2 train_module=model.net \
  trainer.max_epochs=$EP trainer.accelerator=gpu trainer.devices=1 \
  loader.batch_size=128 num_workers=6 loader.num_workers=6 checkpoint_interval=$EP

echo "[greedy2] ==== EVAL DiffusionPolicy ===="
python scripts/plan/eval_wm.py --config-name diffusion_policy_eval_mm \
  policy=dp_greedy2/weights_epoch_$EP.pt cache_dir="$LOCAL_DATASET_DIR" \
  eval.goal_offset_steps=8 eval.num_eval=50 2>&1 | grep -iE "success_rate" | tail -1
echo "[greedy2] ==== EVAL MLP ===="
python scripts/plan/eval_wm.py --config-name diffusion_policy_eval_mm \
  policy=mlp_greedy2/weights_epoch_$EP.pt cache_dir="$LOCAL_DATASET_DIR" \
  eval.goal_offset_steps=8 eval.num_eval=50 2>&1 | grep -iE "success_rate" | tail -1
echo "DONE-GREEDY2"
