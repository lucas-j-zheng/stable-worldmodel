#!/bin/bash
#SBATCH -J ldm-multiseed
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 03:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate
export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

# Multi-seed error bars for ONE dose cell (TAG=mm05|p10, DATA set accordingly).
# TransformerMSE is high-variance across single runs (20..56), so the single-run
# dose sweep was noise-dominated. Run BOTH models at 3 training seeds, eval each
# on the SAME 50 episodes (eval seed fixed in config) -> paired mean+-std. The
# defensible test: diffusion >> tmse on multimodal (mm05), diffusion ~= tmse on
# unimodal (p10).
ENC=lewm_tworoom_mm/weights_epoch_297.pt
EP=${EP:-250}
TAG=${TAG:?set TAG=mm05 or p10}
DATA=${DATA:?set DATA latent name}

run_one () {  # $1=kind(dp|tmse) $2=seed
  local kind=$1 seed=$2 name="${1}_${TAG}_s${2}"
  local extra=""
  if [ "$kind" = tmse ]; then
    extra="model._target_=stable_worldmodel.wm.diffusion_policy.TransformerMSEPolicy \
      ~model.num_diffusion_steps ~model.schedule ~model.num_inference_steps \
      ~model.eta ~model.prediction_type ~model.clip_sample"
  fi
  echo "[ms] train $name ..."
  python scripts/train/train_diffusion_policy.py \
    data=tworoom_latent data.dataset.name=$DATA \
    $extra model.lewm_checkpoint=$ENC output_model_name=$name \
    train_module=model.denoiser seed=$seed \
    trainer.max_epochs=$EP trainer.accelerator=gpu trainer.devices=1 \
    loader.batch_size=128 num_workers=6 loader.num_workers=6 checkpoint_interval=$EP
  echo "[ms] ==== EVAL $name (offset 8, n=50) ===="
  python scripts/plan/eval_wm.py --config-name diffusion_policy_eval_mm \
    policy=${name}/weights_epoch_$EP.pt cache_dir="$LOCAL_DATASET_DIR" \
    eval.goal_offset_steps=8 eval.num_eval=50 2>&1 | grep -iE "success_rate" | tail -1
}

for s in 1 2 3; do
  run_one dp   $s
  run_one tmse $s
done
echo "DONE-MULTISEED-$TAG"
