#!/bin/bash
#SBATCH -J ldm-pusht-cslip-cell
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

# ARC 4 stage 3 (afterok pusht_cslip_prep): THE IRREVERSIBILITY TEST.
# Train diffusion dynamics (v) vs architecture-matched TransformerMSEDynamics on
# contact-slip PushT latents, 3 seeds each, EP=5 (proven pusht budget),
# seed_everything on, matched budget. Closed-loop D-MPC eval on the SAME
# contact-slip env, sweeping DDIM steps @1 vs @20 for diffusion, @1 for TMSE.
#   PRE-REGISTERED: PushT contact is multimodal AND irreversible — the one regime
#   where planning to the conditional mean should fail. If diff@20 > diff@1 here,
#   sampling finally pays and the sampling penalty is bounded to reversible
#   domains. If diff@1 >= diff@20 even here, the penalty is unconditional (sample
#   once, always) and no reachable domain rewards sampling a latent world model.
ENC=lewm_pusht/weights.pt
TAG=${TAG:-cslip05}
SLIP=${SLIP:-0.5}
DATA=${DATA:-pusht_${TAG}_latent.lance}
EP=${EP:-5}

run_train () {  # $1=kind(diff|tmse) $2=seed
  local kind=$1 seed=$2 name="dyn_${1}_pusht_${TAG}_ps${2}"
  local extra="model.prediction_type=v model.num_inference_steps=20"
  if [ "$kind" = tmse ]; then
    extra="model._target_=stable_worldmodel.wm.latent_diffusion.transformer_mse_dynamics.TransformerMSEDynamics"
  fi
  echo "[cslip-cell] train $name (seed $seed) ..."
  python scripts/train/latent_diffusion.py \
    data=pusht_latent data.dataset.name=$DATA \
    $extra model.lewm_checkpoint=$ENC output_model_name=$name \
    seed=$seed \
    trainer.max_epochs=$EP trainer.accelerator=gpu trainer.devices=1 \
    optimizer.lr=3e-4 loader.batch_size=128 num_workers=6 loader.num_workers=6 \
    checkpoint_interval=$EP
}

for s in 1 2 3; do
  run_train diff $s
  run_train tmse $s
  # eval diffusion @1 and @20; TMSE @1 (deterministic single forward)
  for es in 42 123; do
    for st in 1 20; do
      name="dyn_diff_pusht_${TAG}_ps${s}"
      echo "[cslip-cell] ==== EVAL $name steps=$st (n=50, eval seed $es) ===="
      python scripts/plan/eval_wm.py --config-name pusht_diffusion \
        policy=${name}/weights_epoch_$EP.pt cache_dir="$LOCAL_DATASET_DIR" \
        seed=$es world.contact_slip_scale=$SLIP eval.dataset_name=pusht_${TAG}.lance \
        "+model_overrides={num_inference_steps:$st}" 2>&1 \
        | grep -iE "success_rate" | tail -1
    done
    tname="dyn_tmse_pusht_${TAG}_ps${s}"
    echo "[cslip-cell] ==== EVAL $tname (TMSE, n=50, eval seed $es) ===="
    python scripts/plan/eval_wm.py --config-name pusht_diffusion \
      policy=${tname}/weights_epoch_$EP.pt cache_dir="$LOCAL_DATASET_DIR" \
      seed=$es world.contact_slip_scale=$SLIP eval.dataset_name=pusht_${TAG}.lance \
      2>&1 | grep -iE "success_rate" | tail -1
  done
done
echo "DONE-PUSHT-CSLIP-CELL-$TAG"
