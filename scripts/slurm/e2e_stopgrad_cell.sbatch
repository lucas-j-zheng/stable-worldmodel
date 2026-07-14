#!/bin/bash
#SBATCH -J ldm-e2c-cslip
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
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

# ARC 5c — E2E + SIGReg + STOP-GRAD TARGET (the corrected e2e recipe).
# ARC 5/5b both trained with gradients flowing through the prediction-TARGET
# branch of the dynamics loss — a direct "shrink the latents until the target
# is trivial" shortcut (the P2 collapse stack called for stop-grad; it was
# never implemented). 5b's SIGReg then had to FIGHT that gradient, and the
# compromise deleted the multimodality (0.0035, below frozen). Three bundled
# fixes, all targeting the same collapse channel:
#   1. stop-grad target (model.stopgrad_target=true) — encoder learns only
#      through the history/conditioning branch, JEPA-style;
#   2. encoder weight_decay=0 (AdamW decay = gradient-independent shrink);
#   3. batch 128 (SIGReg's statistic scales with B; weight 0.09 was tuned at
#      B=128 in lewm pretraining — the 5b cells ran B=32, ~4x weaker pressure.
#      B=128/EP=5 is also exactly the frozen ARC 2 baseline recipe.)
# PRE-REGISTERED (thresholds match 5b for direct comparison):
#   Q1 anti-collapse: latstats global_std ~1 (5b: pass; ARC5: 0.2/0.04 fail).
#   Q2 shaping survives WITH SIGReg on: e2c-diff latent bimodal >> frozen
#      0.029 (ARC5 no-sigreg: 0.111; 5b: 0.0035 FAIL). Rationale it CAN pass:
#      TwoRoom-slip16 keeps conditional bimodality 0.974 under a SIGReg-trained
#      encoder — conditional modes and a Gaussian marginal are compatible; 5b's
#      deletion was plausibly the collapse-fight compromise stop-grad removes.
#   Q3 planability: base-pusht success off the floor (>= frozen v-diff 10;
#      5b: 1.3-3.0 FAIL). Standing caveat: cslip05-train vs base-eval mismatch.
#   Q4 (P3, iff Q1-Q3 pass): cslip diff@20 vs diff@1 — does sampling finally
#      pay in a multimodality-preserving, plannable latent space?
# Failure reads: Q1 fails -> stop-grad+wd+batch insufficient, collapse is
# deeper than the target-branch shortcut. Q1+Q2 pass, Q3 fails -> the residual
# joint-training deficit is real and independent of collapse (strengthens the
# 5b trilemma). All pass -> the trilemma was a recipe artifact; rerun 5-series
# conclusions before the paper.
KIND=${KIND:?set KIND=diff or tmse}
SEED=${SEED:?set SEED=1,2,3}
TAG=cslip05
SLIP=0.5
EP=${EP:-5}
BS=${BS:-128}
# 5c-aux attribution cell (SIGW=0): ARC 5's 0.111 restoration was measured
# WITHOUT stop-grad, so 5c's Q2 failure is ambiguous between "SIGReg deletes
# modes" and "stop-grad cut the shaping channel". stopgrad-on + SIGReg-off
# distinguishes: bimodal ~0.111 -> SIGReg is the deleter; ~0.005 -> the
# target branch WAS the shaping channel (and the trilemma tightens further).
SIGW=${SIGW:-0.09}
ENC=lewm_pusht/weights.pt
NAME="e2c_${KIND}_${TAG}_ps${SEED}"
[ "$SIGW" != "0.09" ] && NAME="e2c_${KIND}_sw${SIGW/./}_${TAG}_ps${SEED}"

extra="model.prediction_type=v model.num_inference_steps=20"
if [ "$KIND" = tmse ]; then
  extra="model._target_=stable_worldmodel.wm.latent_diffusion.transformer_mse_dynamics.TransformerMSEDynamics"
fi

echo "[e2c-cell] train $NAME from PIXELS (seed $SEED, encoder unfrozen, SIGReg 0.09, stop-grad target, wd0, bs$BS) ..."
python scripts/train/latent_diffusion.py \
  data=pusht data.dataset.name=pusht_${TAG}.lance \
  $extra model.lewm_checkpoint=$ENC model.freeze_lewm=false \
  +model.stopgrad_target=true \
  +encoder_lr_scale=0.1 +sigreg_weight=$SIGW \
  output_model_name=$NAME seed=$SEED \
  trainer.max_epochs=$EP trainer.accelerator=gpu trainer.devices=1 \
  optimizer.lr=3e-4 loader.batch_size=$BS num_workers=5 loader.num_workers=5 \
  checkpoint_interval=$EP

echo "[e2c-cell] re-cache latents with the fine-tuned encoder ..."
LAT=pusht_${TAG}_${NAME}.lance
python scripts/data/cache_latents.py \
  lewm_checkpoint=${NAME}/weights_epoch_$EP.pt +from_dynamics=true \
  source=pusht_${TAG}.lance out_name=$LAT device=cuda batch_size=256

echo "[e2c-cell] Q1 latstats (collapse check; frozen ref std~1) ..."
python scripts/data/latent_marginal_stats.py --dataset $LAT --n-episodes 50 2>&1 | tail -2

echo "[e2c-cell] Q2 LATENT multimodality screen (frozen=0.029, raw=0.981) ..."
python scripts/data/multimodality_diagnostic.py \
  --dataset $LAT --tag ${NAME}_latdyn \
  --mode dynamics --target-col latent --cond-cols latent action \
  --k 96 --n-anchors 4000
python - <<EOF
import json
d = json.load(open('logs/multimodality_${NAME}_latdyn.json'))
print(f"[e2c-cell] >>> ${NAME} LATENT residual_bimodal = "
      f"{d.get('residual_bimodality_fraction', float('nan')):.3f} "
      f"(frozen: 0.029; e2e-no-sigreg diff: 0.111; 5b-sigreg: 0.0035) <<<")
EOF

run_eval () {  # $1=steps(""=det) $2=slip_or_empty $3=label
  local st=$1 slip=$2 label=$3
  local extra_ov=""
  [ -n "$st" ] && extra_ov="+model_overrides={num_inference_steps:$st}"
  local env_ov=""
  [ -n "$slip" ] && env_ov="+world.contact_slip_scale=$slip eval.dataset_name=pusht_${TAG}.lance"
  for es in 42 123; do
    echo "[e2c-cell] ==== EVAL $NAME steps=${st:-det} $label (n=50, eval seed $es) ===="
    local tmp="logs/_e2c_${NAME}_${label}_${st:-det}_${es}.log"
    if python scripts/plan/eval_wm.py --config-name pusht_diffusion \
        policy=${NAME}/weights_epoch_$EP.pt cache_dir="$LOCAL_DATASET_DIR" \
        seed=$es $env_ov $extra_ov > "$tmp" 2>&1; then
      grep -iE "success_rate" "$tmp" | tail -1
    else
      echo "[e2c-cell] EVAL FAILED — traceback tail:"; tail -20 "$tmp"
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
echo "DONE-E2C-$KIND-ps$SEED"
