#!/bin/bash
#SBATCH -J ldm-policy-greedy
#SBATCH -p gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 00:55:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate
export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

# NEGATIVE CONTROL for the thesis: same pipeline as the mm experiment but on the
# GREEDY single-door (UNIMODAL) demos. Prediction: DiffusionPolicy ~= MLP here
# (no multimodality to exploit) -- which, vs the mm result (42% vs 18%), isolates
# multimodality as the cause of the diffusion win. Encoder lewm_tworoom (1-door)
# is in-distribution for the greedy/1-door eval.
ENC=lewm_tworoom/weights_epoch_300.pt
DATA=tworoom_latent_e300.lance
EP=${EP:-250}

echo "[greedy] train DiffusionPolicy ..."
python scripts/train/train_diffusion_policy.py \
  data=tworoom_latent data.dataset.name=$DATA \
  model.lewm_checkpoint=$ENC output_model_name=dp_greedy train_module=model.denoiser \
  trainer.max_epochs=$EP trainer.accelerator=gpu trainer.devices=1 \
  loader.batch_size=128 num_workers=6 loader.num_workers=6 checkpoint_interval=$EP

echo "[greedy] train MLP baseline ..."
python scripts/train/train_diffusion_policy.py \
  data=tworoom_latent data.dataset.name=$DATA \
  model._target_=stable_worldmodel.wm.diffusion_policy.MLPPolicy \
  ~model.denoiser ~model.num_diffusion_steps ~model.schedule \
  ~model.num_inference_steps ~model.eta ~model.prediction_type ~model.clip_sample \
  model.lewm_checkpoint=$ENC output_model_name=mlp_greedy train_module=model.net \
  trainer.max_epochs=$EP trainer.accelerator=gpu trainer.devices=1 \
  loader.batch_size=128 num_workers=6 loader.num_workers=6 checkpoint_interval=$EP

# Eval on 1-door greedy env at the matched goal horizon (offset 8), n=50.
echo "[greedy] ==== EVAL DiffusionPolicy (1-door, unimodal) ===="
python scripts/plan/eval_wm.py --config-name diffusion_policy_eval \
  policy=dp_greedy/weights_epoch_$EP.pt cache_dir="$LOCAL_DATASET_DIR" \
  eval.goal_offset_steps=8 eval.num_eval=50 2>&1 | grep -iE "success_rate" | tail -1
echo "[greedy] ==== EVAL MLP baseline ===="
python scripts/plan/eval_wm.py --config-name diffusion_policy_eval \
  policy=mlp_greedy/weights_epoch_$EP.pt cache_dir="$LOCAL_DATASET_DIR" \
  eval.goal_offset_steps=8 eval.num_eval=50 2>&1 | grep -iE "success_rate" | tail -1
echo "DONE-GREEDY-CONTROL"
