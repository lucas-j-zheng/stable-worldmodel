#!/bin/bash
#SBATCH -J ldm-e2e-warmstart
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --array=0-1

set -uo pipefail
cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate
export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

# ARC 6 / E6.0a — WARM-START vs joint-from-scratch: THE decisive Q3 test
# (design doc 2026-07-14_arc6_e2e_escape_routes.md; diagnosis doc 2026-07-23).
# ARC 5/5b/5c e2e all floored on planability (Q3), and 5c's matched-domain
# CSLIP evals ruled OUT domain shift (D2). Remaining causes: from-scratch
# budget vs cost-blindness vs churn. Here: start from the post-hoc frozen-
# latent x0 model (base-pusht 20-22, the strongest diffusion baseline), then
# gently unfreeze the encoder and fine-tune 2 epochs on base-pusht PIXELS.
# PRE-REGISTERED (P1):
#   parity held (>= ~20 @1)  => floors were a from-scratch budget artifact.
#   drops to floor           => joint training intrinsically harms planability
#                               (headline negative; pivot per diagnosis doc).
#   epoch-1 vs epoch-2 slope => cheap churn evidence (D3c).
# @1-step readout is PRIMARY (universal sampling penalty, arcs 1-3).

SCALES=(0.1 0.03)
LR_SCALE=${SCALES[$SLURM_ARRAY_TASK_ID]}
SNAME=$(echo "ls$LR_SCALE" | tr '.' 'p')   # 0.03 -> ls0p03

SRC=${SRC:-latent_diffusion_pusht_x0}
EP=${EP:-2}
ENC=lewm_pusht/weights.pt
NAME="e2w_x0_base_${SNAME}"

# Newest .pt in the source run (explicit file: folders with several epochs are
# ambiguous for the checkpoint resolver).
CKPT=$(ls -t "$STABLEWM_HOME/checkpoints/$SRC"/*.pt | head -1)
[ -n "$CKPT" ] || { echo "no checkpoint under $SRC"; exit 1; }
INIT="$SRC/$(basename "$CKPT")"
echo "[e2w] ===== warm-start from $INIT, enc_lr_scale=$LR_SCALE, name=$NAME ====="

echo "[e2w] fine-tune $EP epochs from base-pusht PIXELS (encoder unfrozen, SIGReg 0.09) ..."
python scripts/train/latent_diffusion.py \
  data=pusht \
  model.prediction_type=x0 model.num_inference_steps=20 \
  model.lewm_checkpoint=$ENC model.freeze_lewm=false \
  +init_dynamics_from="$INIT" \
  +encoder_lr_scale=$LR_SCALE +sigreg_weight=0.09 \
  output_model_name=$NAME seed=1 \
  trainer.max_epochs=$EP trainer.accelerator=gpu trainer.devices=1 \
  optimizer.lr=3e-4 loader.batch_size=32 num_workers=6 loader.num_workers=6 \
  checkpoint_interval=1

echo "[e2w] re-cache latents with the fine-tuned encoder (epoch $EP) ..."
LAT=pusht_base_${NAME}.lance
python scripts/data/cache_latents.py \
  lewm_checkpoint=${NAME}/weights_epoch_$EP.pt +from_dynamics=true \
  source=pusht_expert_train.lance out_name=$LAT device=cuda batch_size=256

echo "[e2w] Q1 latstats (frozen ref ~0.97; ARC5 collapse 0.2) ..."
python scripts/data/latent_marginal_stats.py --dataset $LAT --n-episodes 50 2>&1 | tail -2

echo "[e2w] Q2 latent multimodality (frozen 0.029 / e2c-sigreg 0.005) ..."
python scripts/data/multimodality_diagnostic.py \
  --dataset $LAT --tag ${NAME}_latdyn \
  --mode dynamics --target-col latent --cond-cols latent action \
  --k 96 --n-anchors 4000 2>&1 | tail -3

# Q3: base-pusht eval per EPOCH (epoch slope = churn evidence) and per
# inference-steps (@1 primary). Frozen x0 baseline: 20-22.
run_eval () {  # $1 = epoch  $2 = steps
  local ep=$1 st=$2
  for es in 42 123; do
    echo "[e2w] ==== EVAL $NAME epoch$ep @${st} steps (base, n=50, eval seed $es) ===="
    local tmp="logs/_e2w_${NAME}_ep${ep}_${st}_${es}.log"
    if python scripts/plan/eval_wm.py --config-name pusht_diffusion \
        policy=${NAME}/weights_epoch_$ep.pt cache_dir="$LOCAL_DATASET_DIR" \
        seed=$es "+model_overrides={num_inference_steps:$st}" > "$tmp" 2>&1; then
      grep -iE "success_rate" "$tmp" | tail -1
    else
      echo "[e2w] EVAL FAILED — traceback tail:"; tail -20 "$tmp"
    fi
  done
}

for ep in $(seq 1 $EP); do
  run_eval $ep 1
  run_eval $ep 20
done
echo "DONE-E2W-$SNAME"
