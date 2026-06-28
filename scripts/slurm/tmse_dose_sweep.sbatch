#!/bin/bash
#SBATCH -J ldm-tmse-dose
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate
export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

# KEYSTONE of Thread A: TransformerMSE (architecture-matched MSE) across the dose.
# Completes the 2x2. Compared to existing DiffusionPolicy (dp_*) per level, the
# Diffusion - TransformerMSE gap is the OBJECTIVE effect with architecture held
# fixed. Thesis predicts: large gap at high measured multimodality (p07/p09,
# ~0.5), ~0 at low (p10, 0.18=unimodal). mm05 tmse already = 20 (vs diff 42).
ENC=lewm_tworoom_mm/weights_epoch_297.pt
EP=${EP:-250}

for TAG in p07 p09 p10; do
  DATA=tworoom_${TAG}_latent.lance
  echo "[tmse-$TAG] train ..."
  python scripts/train/train_diffusion_policy.py \
    data=tworoom_latent data.dataset.name=$DATA \
    model._target_=stable_worldmodel.wm.diffusion_policy.TransformerMSEPolicy \
    ~model.num_diffusion_steps ~model.schedule ~model.num_inference_steps \
    ~model.eta ~model.prediction_type ~model.clip_sample \
    model.lewm_checkpoint=$ENC output_model_name=tmse_${TAG} train_module=model.denoiser \
    trainer.max_epochs=$EP trainer.accelerator=gpu trainer.devices=1 \
    loader.batch_size=128 num_workers=6 loader.num_workers=6 checkpoint_interval=$EP
  echo "[tmse-$TAG] ==== EVAL (offset 8, n=50) ===="
  python scripts/plan/eval_wm.py --config-name diffusion_policy_eval_mm \
    policy=tmse_${TAG}/weights_epoch_$EP.pt cache_dir="$LOCAL_DATASET_DIR" \
    eval.goal_offset_steps=8 eval.num_eval=50 2>&1 | grep -iE "success_rate" | tail -1
done
echo "DONE-TMSE-DOSE"
