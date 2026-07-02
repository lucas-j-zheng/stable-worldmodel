#!/bin/bash
#SBATCH -J ldm-slip-x0
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 05:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate
export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

# Mechanism rung 2 for the training-signal advantage (1-step probe showed the
# diffusion gap is a TRAINING effect: 1 forward ~ 20 refinements, both >> TMSE).
# Separate the two candidate ingredients:
#   noised-input training (q_sample noise curriculum)  vs  v-param target.
# Train the same denoiser with prediction_type=x0 (denoising regression of the
# clean target — keeps the noise curriculum, drops the v-target), 3 seeds, then
# closed-loop eval @1 step and @default.
#   x0 ~ v (>> TMSE)  => the NOISE CURRICULUM is the ingredient
#   x0 ~ TMSE (<< v)  => the v-parametrization target is the ingredient
ENC=lewm_tworoom/weights_epoch_300.pt
EP=100
TAG=${TAG:?set TAG=slip8 or slip0}
SLIP=${SLIP:?set SLIP=8 or 0}
DATA=tworoom_${TAG}_latent.lance

for s in 1 2 3; do
  name="dyn_x0_${TAG}_ps${s}"
  echo "[x0cell] train $name (seed $s) ..."
  python scripts/train/latent_diffusion.py \
    data=tworoom_latent data.dataset.name=$DATA \
    model.prediction_type=x0 model.num_inference_steps=20 \
    model.lewm_checkpoint=$ENC output_model_name=$name \
    seed=$s \
    trainer.max_epochs=$EP trainer.accelerator=gpu trainer.devices=1 \
    optimizer.lr=3e-4 loader.batch_size=128 num_workers=6 loader.num_workers=6 \
    checkpoint_interval=$EP
  for es in 42 123; do
    for st in 1 20; do
      echo "[x0cell] ==== EVAL $name steps=$st (offset 25, n=50, eval seed $es) ===="
      python scripts/plan/eval_wm.py --config-name tworoom_slip_diffusion \
        policy=${name}/weights_epoch_$EP.pt cache_dir="$LOCAL_DATASET_DIR" \
        seed=$es world.slip_scale=$SLIP eval.dataset_name=tworoom_${TAG}.lance \
        "+model_overrides={num_inference_steps:$st}" 2>&1 \
        | grep -iE "success_rate" | tail -1
    done
  done
done
echo "DONE-X0-CELL-$TAG"
