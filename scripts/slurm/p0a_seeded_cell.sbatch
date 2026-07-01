#!/bin/bash
#SBATCH -J ldm-p0a-seeded
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 05:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate
export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

# P0a seeded endpoint cell (TAG=mm05|p10). Fixes the two audit flags in one job:
#   1. REAL fixed seeds — train_diffusion_policy.py now calls seed_everything;
#      the earlier "3-seed" runs (dp_*_s1..3 / tmse_*_s1..3) were unseeded.
#   2. n=100 per (model, train-seed): eval at TWO eval seeds (42, 123), n=50
#      each — also un-fixes episode-sampling noise that eval seed 42 froze out.
# Each eval now dumps positions_off8_seed{ES}.npz (per-step agent positions +
# successes + goal positions) — the failure-mode probe: does TMSE fail AT THE
# WALL between the doors on mm05? Checkpoints use _ps{seed} names to avoid
# clobbering the old unseeded runs.
ENC=lewm_tworoom_mm/weights_epoch_297.pt
EP=${EP:-250}
TAG=${TAG:?set TAG=mm05 or p10}
DATA=${DATA:?set DATA latent name}

run_one () {  # $1=kind(dp|tmse) $2=seed
  local kind=$1 seed=$2 name="${1}_${TAG}_ps${2}"
  local extra=""
  if [ "$kind" = tmse ]; then
    extra="model._target_=stable_worldmodel.wm.diffusion_policy.TransformerMSEPolicy \
      ~model.num_diffusion_steps ~model.schedule ~model.num_inference_steps \
      ~model.eta ~model.prediction_type ~model.clip_sample"
  fi
  echo "[p0a] train $name (seed $seed) ..."
  python scripts/train/train_diffusion_policy.py \
    data=tworoom_latent data.dataset.name=$DATA \
    $extra model.lewm_checkpoint=$ENC output_model_name=$name \
    train_module=model.denoiser seed=$seed \
    trainer.max_epochs=$EP trainer.accelerator=gpu trainer.devices=1 \
    loader.batch_size=128 num_workers=6 loader.num_workers=6 checkpoint_interval=$EP
  for es in 42 123; do
    echo "[p0a] ==== EVAL $name (offset 8, n=50, eval seed $es) ===="
    python scripts/plan/eval_wm.py --config-name diffusion_policy_eval_mm \
      policy=${name}/weights_epoch_$EP.pt cache_dir="$LOCAL_DATASET_DIR" \
      seed=$es eval.goal_offset_steps=8 eval.num_eval=50 2>&1 \
      | grep -iE "success_rate|positions saved" | tail -2
  done
}

for s in ${SEEDS:-1 2 3}; do
  run_one dp   $s
  run_one tmse $s
done
echo "DONE-P0A-$TAG"
