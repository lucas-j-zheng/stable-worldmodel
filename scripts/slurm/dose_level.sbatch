#!/bin/bash
#SBATCH -J ldm-dose
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

# One multimodality DOSE level. PROB = P(door 0); 0.5 bimodal .. 1.0 unimodal.
# TAG labels datasets/checkpoints. Same 2-door geometry + encoder as the p=0.5
# experiment, so only the demo mode-mixing changes -> clean dose axis.
PROB=${PROB:?set PROB}
TAG=${TAG:?set TAG}
ENC=lewm_tworoom_mm/weights_epoch_297.pt
EP=${EP:-250}

echo "[dose $TAG] collect (door_prob=$PROB) ..."
python scripts/data/collect_tworooms_multimodal.py cache_dir="$STABLEWM_HOME" \
  num_traj=6000 world.num_envs=10 world.max_episode_steps=100 \
  +stochastic_door=true +door_prob=$PROB +out_name=tworoom_${TAG}.lance

echo "[dose $TAG] cache latents ..."
python scripts/data/cache_latents.py lewm_checkpoint=$ENC \
  source=tworoom_${TAG}.lance out_name=tworoom_${TAG}_latent.lance device=cuda batch_size=256

echo "[dose $TAG] train DiffusionPolicy ..."
python scripts/train/train_diffusion_policy.py \
  data=tworoom_latent data.dataset.name=tworoom_${TAG}_latent.lance \
  model.lewm_checkpoint=$ENC output_model_name=dp_${TAG} train_module=model.denoiser \
  trainer.max_epochs=$EP trainer.accelerator=gpu trainer.devices=1 \
  loader.batch_size=128 num_workers=6 loader.num_workers=6 checkpoint_interval=$EP

echo "[dose $TAG] train MLP ..."
python scripts/train/train_diffusion_policy.py \
  data=tworoom_latent data.dataset.name=tworoom_${TAG}_latent.lance \
  model._target_=stable_worldmodel.wm.diffusion_policy.MLPPolicy \
  ~model.denoiser ~model.num_diffusion_steps ~model.schedule \
  ~model.num_inference_steps ~model.eta ~model.prediction_type ~model.clip_sample \
  model.lewm_checkpoint=$ENC output_model_name=mlp_${TAG} train_module=model.net \
  trainer.max_epochs=$EP trainer.accelerator=gpu trainer.devices=1 \
  loader.batch_size=128 num_workers=6 loader.num_workers=6 checkpoint_interval=$EP

# Eval both on the SAME 2-door env/init-goal (tworoom_mm0) so only the policy
# differs across the dose curve. goal_offset 8 (matched), n=50.
echo "[dose $TAG] ==== EVAL DiffusionPolicy ===="
python scripts/plan/eval_wm.py --config-name diffusion_policy_eval_mm \
  policy=dp_${TAG}/weights_epoch_$EP.pt cache_dir="$LOCAL_DATASET_DIR" \
  eval.goal_offset_steps=8 eval.num_eval=50 2>&1 | grep -iE "success_rate" | tail -1
echo "[dose $TAG] ==== EVAL MLP ===="
python scripts/plan/eval_wm.py --config-name diffusion_policy_eval_mm \
  policy=mlp_${TAG}/weights_epoch_$EP.pt cache_dir="$LOCAL_DATASET_DIR" \
  eval.goal_offset_steps=8 eval.num_eval=50 2>&1 | grep -iE "success_rate" | tail -1
echo "DONE-DOSE-$TAG (prob=$PROB)"
