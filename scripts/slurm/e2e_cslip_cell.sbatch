#!/bin/bash
#SBATCH -J ldm-e2e-cslip
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate
export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

# ARC 5 — E2E: train the ENCODER jointly with the dynamics objective.
# ARC 4 capstone: the frozen lewm_pusht encoder collapses stochastic-contact
# multimodality (raw 0.98 -> latent 0.029), so no frozen-latent experiment can
# ever hand a generative model multimodal dynamics. This cell unfreezes the
# encoder (fine-tune at 0.1x LR from the pretrained lewm) and trains it jointly
# with (a) the v-diffusion denoising objective or (b) the matched clean-MSE
# objective, FROM PIXELS on the contact-slip dataset.
# PRE-REGISTERED:
#   P1 (denoising-as-latent-shaping): e2e-diff latent bimodality >> 0.029 —
#      a distribution-capable loss can afford to keep the modes.
#   P2 (MSE predictability pressure): e2e-tmse latent bimodality stays ~0 (or
#      drops) — an MSE dynamics loss REWARDS the encoder for deleting
#      unpredictable structure.
#   P3: if P1 holds, the @1-vs-@20 eval on e2e-diff is the FIRST fair test of
#      sampling (latents finally carry the branching); @20 > @1 would be the
#      program's first genuine sampling win.
# Readout order: (1) latent bimodality per arm (the latent-shaping verdict),
# (2) closed-loop D-MPC on the contact-slip env.
KIND=${KIND:?set KIND=diff or tmse}
SEED=${SEED:?set SEED=1,2,3}
TAG=cslip05
SLIP=0.5
EP=${EP:-5}
ENC=lewm_pusht/weights.pt
NAME="e2e_${KIND}_${TAG}_ps${SEED}"

extra="model.prediction_type=v model.num_inference_steps=20"
if [ "$KIND" = tmse ]; then
  extra="model._target_=stable_worldmodel.wm.latent_diffusion.transformer_mse_dynamics.TransformerMSEDynamics"
fi

echo "[e2e-cell] train $NAME from PIXELS (seed $SEED, encoder unfrozen) ..."
python scripts/train/latent_diffusion.py \
  data=pusht data.dataset.name=pusht_${TAG}.lance \
  $extra model.lewm_checkpoint=$ENC model.freeze_lewm=false \
  +encoder_lr_scale=0.1 \
  output_model_name=$NAME seed=$SEED \
  trainer.max_epochs=$EP trainer.accelerator=gpu trainer.devices=1 \
  optimizer.lr=3e-4 loader.batch_size=32 num_workers=6 loader.num_workers=6 \
  checkpoint_interval=$EP

echo "[e2e-cell] re-cache latents with the FINE-TUNED encoder ..."
LAT=pusht_${TAG}_${NAME}.lance
python scripts/data/cache_latents.py \
  lewm_checkpoint=${NAME}/weights_epoch_$EP.pt +from_dynamics=true \
  source=pusht_${TAG}.lance out_name=$LAT device=cuda batch_size=256

echo "[e2e-cell] LATENT multimodality screen (KEY READOUT; frozen enc = 0.029) ..."
python scripts/data/multimodality_diagnostic.py \
  --dataset $LAT --tag ${NAME}_latdyn \
  --mode dynamics --target-col latent --cond-cols latent action \
  --k 96 --n-anchors 4000
python - <<EOF
import json
d = json.load(open('logs/multimodality_${NAME}_latdyn.json'))
print(f"[e2e-cell] >>> ${NAME} LATENT residual_bimodal = "
      f"{d.get('residual_bimodality_fraction', float('nan')):.3f} "
      f"(frozen encoder: 0.029; raw state: 0.981) <<<")
EOF

STEPS_LIST="1 20"
[ "$KIND" = tmse ] && STEPS_LIST="1"
for es in 42 123; do
  for st in $STEPS_LIST; do
    echo "[e2e-cell] ==== EVAL $NAME steps=$st (n=50, eval seed $es) ===="
    python scripts/plan/eval_wm.py --config-name pusht_diffusion \
      policy=${NAME}/weights_epoch_$EP.pt cache_dir="$LOCAL_DATASET_DIR" \
      seed=$es +world.contact_slip_scale=$SLIP eval.dataset_name=pusht_${TAG}.lance \
      "+model_overrides={num_inference_steps:$st}" 2>&1 \
      | grep -iE "success_rate" | tail -1
  done
done
echo "DONE-E2E-CSLIP-$KIND-ps$SEED"
