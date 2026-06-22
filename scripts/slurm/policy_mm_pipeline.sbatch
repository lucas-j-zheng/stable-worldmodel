#!/bin/bash
#SBATCH -J ldm-policy-mm
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
mkdir -p logs

ENC=lewm_tworoom_mm/weights_epoch_300.pt
EP=${EP:-250}

# 1. Cache mm latents with the 2-door encoder (in-distribution).
echo "[policy-mm] cache latents ..."
python scripts/data/cache_latents.py lewm_checkpoint=$ENC \
  source=tworoom_mm0.lance out_name=tworoom_mm_latent2.lance device=cuda batch_size=256

# 2. Train the diffusion policy and the MSE baseline on the same mm latents.
echo "[policy-mm] train DiffusionPolicy ..."
python scripts/train/train_diffusion_policy.py \
  data=tworoom_latent data.dataset.name=tworoom_mm_latent2.lance \
  model.lewm_checkpoint=$ENC output_model_name=dp_mm train_module=model.denoiser \
  trainer.max_epochs=$EP trainer.accelerator=gpu trainer.devices=1 \
  loader.batch_size=128 num_workers=6 loader.num_workers=6 checkpoint_interval=$EP

echo "[policy-mm] train MLP baseline ..."
python scripts/train/train_diffusion_policy.py \
  data=tworoom_latent data.dataset.name=tworoom_mm_latent2.lance \
  model._target_=stable_worldmodel.wm.diffusion_policy.MLPPolicy \
  ~model.denoiser ~model.num_diffusion_steps ~model.schedule \
  ~model.num_inference_steps ~model.eta ~model.prediction_type ~model.clip_sample \
  model.lewm_checkpoint=$ENC output_model_name=mlp_mm train_module=model.net \
  trainer.max_epochs=$EP trainer.accelerator=gpu trainer.devices=1 \
  loader.batch_size=128 num_workers=6 loader.num_workers=6 checkpoint_interval=$EP

# 3. THE DECISIVE TEST: closed-loop on 2-door env, diffusion policy vs MSE.
echo "[policy-mm] ==== EVAL DiffusionPolicy ===="
python scripts/plan/eval_wm.py --config-name diffusion_policy_eval_mm \
  policy=dp_mm/weights_epoch_$EP.pt cache_dir="$LOCAL_DATASET_DIR"
echo "[policy-mm] ==== EVAL MLP baseline ===="
python scripts/plan/eval_wm.py --config-name diffusion_policy_eval_mm \
  policy=mlp_mm/weights_epoch_$EP.pt cache_dir="$LOCAL_DATASET_DIR"

echo "==== DIFFUSION POLICY ===="
grep -i metrics "$STABLEWM_HOME/checkpoints/dp_mm/diffusion_policy_eval_mm_results.txt" | tail -1
echo "==== MLP BASELINE ===="
grep -i metrics "$STABLEWM_HOME/checkpoints/mlp_mm/diffusion_policy_eval_mm_results.txt" | tail -1
