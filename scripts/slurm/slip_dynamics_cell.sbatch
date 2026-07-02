#!/bin/bash
#SBATCH -J ldm-slip-dyncell
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

# P1 Route 1 stage 3 (chained after slip_cache_latents via --dependency):
# train diffusion DYNAMICS vs architecture-matched TransformerMSEDynamics on
# ONE slip cell (TAG=slip8 or slip0-control), 3 seeds each, seed_everything on.
# Both models post-hoc on the same cached latents + same budget (this also
# fixes the old training-budget-parity confound). Closed-loop D-MPC eval on
# the slip env is the NEXT stage (needs the eval config; separate job).
# Lessons baked in from P0a: real seeds, expect high cross-run variance,
# never read open-loop MSE as the verdict.
ENC=lewm_tworoom/weights_epoch_300.pt
EP=${EP:-250}
TAG=${TAG:?set TAG=slip8 or slip0}
DATA=${DATA:?set DATA latent lance name}

# Gate: the LATENT-space screen from the cache stage must confirm bimodality
# for slip8 (control cell slip0 exempt — it SHOULD be unimodal).
if [ "$TAG" != slip0 ]; then
python - <<EOF
import json, sys
d = json.load(open('logs/multimodality_${TAG}_latdyn.json'))
frac = d.get('residual_bimodality_fraction', 0.0)
print(f'[dyncell] gate: ${TAG} latent residual_bimodal = {frac:.3f}')
sys.exit(0 if frac > 0.3 else 1)
EOF
fi

run_one () {  # $1=kind(diff|tmse) $2=seed
  local kind=$1 seed=$2 name="dyn_${1}_${TAG}_ps${2}"
  local extra="model.prediction_type=v model.num_inference_steps=20"
  if [ "$kind" = tmse ]; then
    extra="model._target_=stable_worldmodel.wm.latent_diffusion.transformer_mse_dynamics.TransformerMSEDynamics"
  fi
  echo "[dyncell] train $name (seed $seed) ..."
  python scripts/train/latent_diffusion.py \
    data=tworoom_latent data.dataset.name=$DATA \
    $extra model.lewm_checkpoint=$ENC output_model_name=$name \
    seed=$seed \
    trainer.max_epochs=$EP trainer.accelerator=gpu trainer.devices=1 \
    optimizer.lr=3e-4 loader.batch_size=128 num_workers=6 loader.num_workers=6 \
    checkpoint_interval=$EP
}

for s in 1 2 3; do
  run_one diff $s
  run_one tmse $s
done
echo "DONE-SLIP-DYNCELL-$TAG"
