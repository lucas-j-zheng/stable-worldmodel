#!/bin/bash
#SBATCH -J ldm-pusht-recipe
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate
export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

# ARC 2: does the NOISE-CURRICULUM RECIPE transfer to PushT (contact
# manipulation, deterministic physics)? TwoRoom verdict: x0@1step beats
# clean-MSE by +10-14 regardless of multimodality; sampling neutral-to-harmful.
# If it transfers, the recipe is a general claim AND likely overturns the old
# "diffusion loses on PushT" negative (which carried the budget confound +
# the 20-step sampling penalty).
# KIND=x0|v|tmse — one objective per job, 3 real seeds, matched budget (EP=5,
# the proven-to-fit PushT budget), closed-loop eval @1 and @20 steps.
ENC=lewm_pusht/weights_epoch_300.pt
EP=${EP:-5}
KIND=${KIND:?set KIND=x0, v, or tmse}

for s in 1 2 3; do
  name="dyn_${KIND}_pusht_ps${s}"
  extra="model.prediction_type=${KIND} model.num_inference_steps=20"
  if [ "$KIND" = tmse ]; then
    extra="model._target_=stable_worldmodel.wm.latent_diffusion.transformer_mse_dynamics.TransformerMSEDynamics"
  fi
  echo "[pusht-recipe] train $name (seed $s) ..."
  python scripts/train/latent_diffusion.py \
    data=pusht_latent data.dataset.name=pusht_latent.lance \
    $extra model.lewm_checkpoint=$ENC output_model_name=$name \
    seed=$s \
    trainer.max_epochs=$EP trainer.accelerator=gpu trainer.devices=1 \
    optimizer.lr=3e-4 loader.batch_size=128 num_workers=6 loader.num_workers=6 \
    checkpoint_interval=$EP

  STEPS_LIST="1 20"
  [ "$KIND" = tmse ] && STEPS_LIST="1"
  for es in 42 123; do
    for st in $STEPS_LIST; do
      echo "[pusht-recipe] ==== EVAL $name steps=$st (n=50, eval seed $es) ===="
      python scripts/plan/eval_wm.py --config-name pusht_diffusion \
        policy=${name}/weights_epoch_$EP.pt cache_dir="$LOCAL_DATASET_DIR" \
        seed=$es "+model_overrides={num_inference_steps:$st}" 2>&1 \
        | grep -iE "success_rate" | tail -1
    done
  done
done
echo "DONE-PUSHT-RECIPE-$KIND"
