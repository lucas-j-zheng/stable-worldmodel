#!/bin/bash
#SBATCH -J ldm-e2s-cslip
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -uo pipefail
cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate
export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

# ARC 5b — E2E + SIGReg: fix the diagnosed collapse. ARC 5 showed (a) the
# denoising objective partially restores contact multimodality in latent space
# (0.111 vs frozen 0.029) while MSE doesn't (0.021), BUT (b) bare-dynamics-loss
# fine-tuning COLLAPSES the representation (global_std 0.97 -> 0.2 diff / 0.04
# tmse) and floors closed-loop planning everywhere. This cell retains the SIGReg
# anti-collapse regularizer (weight 0.09, as in lewm pretraining) during joint
# training. PRE-REGISTERED:
#   Q1 anti-collapse: latstats global_std stays ~1 (vs 0.2/0.04 without).
#   Q2 shaping survives: e2s-diff latent bimodal >> frozen 0.029 with SIGReg on.
#   Q3 planability: base-pusht success comes off the floor (>= frozen v-diff 10).
#   Q4 (P3, finally answerable iff Q1-Q3 pass): cslip diff@20 vs diff@1 — does
#      sampling pay in a multimodality-preserving, PLANNABLE latent space?
KIND=${KIND:?set KIND=diff or tmse}
SEED=${SEED:?set SEED=1,2,3}
TAG=cslip05
SLIP=0.5
EP=${EP:-5}
ENC=lewm_pusht/weights.pt
NAME="e2s_${KIND}_${TAG}_ps${SEED}"

extra="model.prediction_type=v model.num_inference_steps=20"
if [ "$KIND" = tmse ]; then
  extra="model._target_=stable_worldmodel.wm.latent_diffusion.transformer_mse_dynamics.TransformerMSEDynamics"
fi

echo "[e2s-cell] train $NAME from PIXELS (seed $SEED, encoder unfrozen, SIGReg 0.09) ..."
python scripts/train/latent_diffusion.py \
  data=pusht data.dataset.name=pusht_${TAG}.lance \
  $extra model.lewm_checkpoint=$ENC model.freeze_lewm=false \
  +encoder_lr_scale=0.1 +sigreg_weight=0.09 \
  output_model_name=$NAME seed=$SEED \
  trainer.max_epochs=$EP trainer.accelerator=gpu trainer.devices=1 \
  optimizer.lr=3e-4 loader.batch_size=32 num_workers=6 loader.num_workers=6 \
  checkpoint_interval=$EP

echo "[e2s-cell] re-cache latents with the SIGReg-fine-tuned encoder ..."
LAT=pusht_${TAG}_${NAME}.lance
python scripts/data/cache_latents.py \
  lewm_checkpoint=${NAME}/weights_epoch_$EP.pt +from_dynamics=true \
  source=pusht_${TAG}.lance out_name=$LAT device=cuda batch_size=256

echo "[e2s-cell] Q1 latstats (collapse check; frozen ref std~1) ..."
python scripts/data/latent_marginal_stats.py --dataset $LAT --n-episodes 50 2>&1 | tail -2

echo "[e2s-cell] Q2 LATENT multimodality screen (frozen=0.029, raw=0.981) ..."
python scripts/data/multimodality_diagnostic.py \
  --dataset $LAT --tag ${NAME}_latdyn \
  --mode dynamics --target-col latent --cond-cols latent action \
  --k 96 --n-anchors 4000
python - <<EOF
import json
d = json.load(open('logs/multimodality_${NAME}_latdyn.json'))
print(f"[e2s-cell] >>> ${NAME} LATENT residual_bimodal = "
      f"{d.get('residual_bimodality_fraction', float('nan')):.3f} "
      f"(frozen: 0.029; e2e-no-sigreg diff: 0.111 / tmse: 0.021) <<<")
EOF

run_eval () {  # $1=steps(""=det) $2=slip_or_empty $3=label
  local st=$1 slip=$2 label=$3
  local extra_ov=""
  [ -n "$st" ] && extra_ov="+model_overrides={num_inference_steps:$st}"
  local env_ov=""
  [ -n "$slip" ] && env_ov="+world.contact_slip_scale=$slip eval.dataset_name=pusht_${TAG}.lance"
  for es in 42 123; do
    echo "[e2s-cell] ==== EVAL $NAME steps=${st:-det} $label (n=50, eval seed $es) ===="
    local tmp="logs/_e2s_${NAME}_${label}_${st:-det}_${es}.log"
    if python scripts/plan/eval_wm.py --config-name pusht_diffusion \
        policy=${NAME}/weights_epoch_$EP.pt cache_dir="$LOCAL_DATASET_DIR" \
        seed=$es $env_ov $extra_ov > "$tmp" 2>&1; then
      grep -iE "success_rate" "$tmp" | tail -1
    else
      echo "[e2s-cell] EVAL FAILED — traceback tail:"; tail -20 "$tmp"
    fi
  done
}

# Q3: base pusht (frozen baselines exist). Q4: cslip env.
if [ "$KIND" = diff ]; then
  run_eval 1  ""      BASE
  run_eval 20 ""      BASE
  run_eval 1  "$SLIP" CSLIP
  run_eval 20 "$SLIP" CSLIP
else
  run_eval "" ""      BASE
  run_eval "" "$SLIP" CSLIP
fi
echo "DONE-E2S-$KIND-ps$SEED"
