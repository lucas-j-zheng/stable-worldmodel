#!/bin/bash
#SBATCH -J ldm-tmse
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

# THE MISSING CONTROL: same transformer architecture as DiffusionPolicy, but MSE
# objective (deterministic). Trained on the SAME mm latents as dp_mm/mlp_mm.
# 3-way comparison isolates objective from architecture:
#   DiffusionPolicy (transformer + diffusion):  42% (have)
#   TransformerMSE  (transformer + MSE):         ??? (this job)
#   MLP             (MLP + MSE):                 18% (have)
# tmse ~ 42 -> architecture (transformer) explains the win, NOT diffusion.
# tmse ~ 18 -> objective (diffusion/multimodality) explains it -> thesis holds.
ENC=lewm_tworoom_mm/weights_epoch_297.pt
DATA=tworoom_mm_latent2.lance
EP=${EP:-250}

echo "[tmse] train TransformerMSE (same denoiser arch, MSE objective) ..."
python scripts/train/train_diffusion_policy.py \
  data=tworoom_latent data.dataset.name=$DATA \
  model._target_=stable_worldmodel.wm.diffusion_policy.TransformerMSEPolicy \
  ~model.num_diffusion_steps ~model.schedule ~model.num_inference_steps \
  ~model.eta ~model.prediction_type ~model.clip_sample \
  model.lewm_checkpoint=$ENC output_model_name=tmse_mm train_module=model.denoiser \
  trainer.max_epochs=$EP trainer.accelerator=gpu trainer.devices=1 \
  loader.batch_size=128 num_workers=6 loader.num_workers=6 checkpoint_interval=$EP

# Eval at the two matched offsets, n=50, on the same 2-door mm env.
for GO in 8 12; do
  echo "[tmse] ==== EVAL TransformerMSE goal_offset=$GO ===="
  python scripts/plan/eval_wm.py --config-name diffusion_policy_eval_mm \
    policy=tmse_mm/weights_epoch_$EP.pt cache_dir="$LOCAL_DATASET_DIR" \
    eval.goal_offset_steps=$GO eval.num_eval=50 2>&1 | grep -iE "success_rate" | tail -1
done
echo "DONE-TMSE"
