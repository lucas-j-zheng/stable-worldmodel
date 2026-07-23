#!/bin/bash
#SBATCH -J ldm-e2s-pareto
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --array=0-4

set -uo pipefail
cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate
export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

# ARC 6 / E6.1 — sigreg_weight PARETO. Maps the interior of the ARC 5b trilemma.
# We only have the two endpoints: w=0 (collapse: global_std ~0.2, latent bimodal
# 0.111, floors) and w=0.09 (calibrated: std ~0.9, bimodal 0.007, still floors).
# This sweep asks: is there a weight where global_std is high enough to plan AND
# latent bimodality stays elevated (>> frozen 0.029), or are they strictly
# anti-correlated (trilemma confirmed as a curve, not just endpoints)?
#
# PRE-REGISTERED (design doc 2026-07-14_arc6_e2e_escape_routes.md, E6.1):
#   (1) global_std monotone INCREASING in weight (~0.2 -> ~0.9).
#   (2) latent bimodality monotone DECREASING in weight (0.111 -> 0.007).
#   (3) interior sweet spot (std>=0.6 AND bimodal>=0.05) IF any slack exists;
#       strict anti-correlation with no such point => trilemma CONFIRMED as a
#       curve => abandon marginal-shaping, go to E6.2/E6.4.
#   (4) base-pusht planability likely floored at ALL weights (Q3 is a separate
#       axis, bucket 0); coming OFF the floor at some weight is an unplanned
#       positive worth chasing immediately.
#
# diff objective, seed 1 (single-seed MAP; add seeds only at a chosen weight).
# Endpoints 0.0 / 0.09 re-run to anchor harness consistency vs ARC 5/5b.

WEIGHTS=(0.0 0.02 0.045 0.09 0.18)
W=${WEIGHTS[$SLURM_ARRAY_TASK_ID]}
WNAME=$(echo "w$W" | tr '.' 'p')          # 0.045 -> w0p045 (dot-free dir name)

KIND=diff
SEED=1
TAG=cslip05
EP=${EP:-5}
ENC=lewm_pusht/weights.pt
NAME="e2s_${KIND}_${TAG}_${WNAME}_ps${SEED}"

echo "[pareto] ===== weight=$W  name=$NAME  (array task $SLURM_ARRAY_TASK_ID) ====="

echo "[pareto] train $NAME from PIXELS (encoder unfrozen, SIGReg weight=$W) ..."
python scripts/train/latent_diffusion.py \
  data=pusht data.dataset.name=pusht_${TAG}.lance \
  model.prediction_type=v model.num_inference_steps=20 \
  model.lewm_checkpoint=$ENC model.freeze_lewm=false \
  +encoder_lr_scale=0.1 +sigreg_weight=$W \
  output_model_name=$NAME seed=$SEED \
  trainer.max_epochs=$EP trainer.accelerator=gpu trainer.devices=1 \
  optimizer.lr=3e-4 loader.batch_size=32 num_workers=6 loader.num_workers=6 \
  checkpoint_interval=$EP

echo "[pareto] re-cache latents with the fine-tuned encoder ..."
LAT=pusht_${TAG}_${NAME}.lance
python scripts/data/cache_latents.py \
  lewm_checkpoint=${NAME}/weights_epoch_$EP.pt +from_dynamics=true \
  source=pusht_${TAG}.lance out_name=$LAT device=cuda batch_size=256

echo "[pareto] Q1 latstats (global_std; frozen ref ~0.97, collapse ~0.2) ..."
python scripts/data/latent_marginal_stats.py --dataset $LAT --n-episodes 50 2>&1 | tail -2

echo "[pareto] Q2 LATENT multimodality screen (frozen=0.029, w0=0.111, w0.09=0.007) ..."
python scripts/data/multimodality_diagnostic.py \
  --dataset $LAT --tag ${NAME}_latdyn \
  --mode dynamics --target-col latent --cond-cols latent action \
  --k 96 --n-anchors 4000
python - <<EOF
import json
d = json.load(open('logs/multimodality_${NAME}_latdyn.json'))
print(f"[pareto] >>> weight=$W  latent residual_bimodal = "
      f"{d.get('residual_bimodality_fraction', float('nan')):.4f} "
      f"(frozen 0.029 / w0 0.111 / w0.09 0.007) <<<")
EOF

# Q3 readout (base-pusht only; frozen baselines v-diff 10 / x0 20-22 / MSE 30.7).
run_base_eval () {  # $1 = num_inference_steps
  local st=$1
  for es in 42 123; do
    echo "[pareto] ==== EVAL $NAME @${st} steps (base-pusht, n=50, eval seed $es) ===="
    local tmp="logs/_pareto_${NAME}_base_${st}_${es}.log"
    if python scripts/plan/eval_wm.py --config-name pusht_diffusion \
        policy=${NAME}/weights_epoch_$EP.pt cache_dir="$LOCAL_DATASET_DIR" \
        seed=$es "+model_overrides={num_inference_steps:$st}" > "$tmp" 2>&1; then
      grep -iE "success_rate" "$tmp" | tail -1
    else
      echo "[pareto] EVAL FAILED — traceback tail:"; tail -20 "$tmp"
    fi
  done
}

run_base_eval 1
run_base_eval 20

echo "DONE-E2S-PARETO-w${W}"
