#!/bin/bash
#SBATCH -J ldm-e6p7-bn
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 10:00:00
#SBATCH -x gpu3005
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --array=0-2

# ============================================================================
# E6.7 — does the BN train/eval mismatch explain the ENTIRE e2e deficit?
#
# THE BUG. lewm.encode runs the CLS token through `projector`, whose MLP carries
# a BatchNorm1d. LatentDiffusionDynamics.train() forced lewm.eval() only when
# the encoder was FROZEN. So on the e2e path the encoder trained in train mode
# (BN centering each batch) and planned in eval mode (BN running stats).
# Measured on the same 1090 frames (4606058):
#     train mode  mean_norm 0.139  within_std 0.656
#     eval  mode  mean_norm 3.554  within_std 1.016
# The denoiser was fit on batch-centered latents and asked to plan on
# running-stat latents -- a large mean shift plus a ~55% scale change, present
# ONLY on the e2e path. The frozen pipeline never had it.
#
# This is a candidate cause for EVERY e2e result in the program: the ARC 5/5b/5c
# planning floors, the E6.0a warm-start halving, the E6.1 Pareto floors, and
# possibly the Q2 mode deletion and the E6.6 agent-blindness (both measured in
# eval mode on encoders trained in train mode).
#
# THE FIX (default ON as of this commit): pin the encoder's BN layers to
# inference mode while leaving weights trainable and dropout untouched, so
# normalization statistics are the single changed variable.
#
# ARMS -- each mirrors a banked configuration EXACTLY, changing nothing else:
#   0  warm-start ls0.03, base pusht, 2 epochs   banked @1: 10 / 14
#   1  warm-start ls0.1,  base pusht, 2 epochs   banked @1:  8 /  6
#   2  from-scratch e2c (v-pred, cslip05, 5 ep)  banked @1 base: 0 / 4
#   Frozen x0 reference (unaffected by the bug): 20-22 @1.
#   The banked numbers are the control: same configs, same seeds, same env.
#
# PRE-REGISTERED READS
#   Arm 0 recovers toward 20  =>  the warm-start deficit WAS the normalization
#     mismatch. The "joint fine-tuning degrades planning ~2x" verdict is
#     retracted, and the e2e route reopens -- consistent with the reference
#     implementation (arXiv 2603.19312) reaching 96% on PushT WITH e2e.
#   Arm 0 stays 10-14  =>  BN was not the mechanism; the e2e deficit stands and
#     E6.6 agent-blindness remains the lead. This is the pre-registered null.
#   Arm 2 lifts off the 0-6 floor  =>  the ARC 5/5b/5c from-scratch floors are
#     ALSO explained, and the whole e2e arc needs re-running rather than
#     reinterpreting.
#   Q2 (residual_bimodal, frozen 0.029, every e2e cell so far <=0.01): a rise
#     toward 0.029 means "e2e deletes modes" was an artifact of measuring an
#     encoder in a mode it was never trained in => E6.1's KILL is void and the
#     Pareto must be redone.
#   Latent geometry: latstats now reports within_std. Under the fix, training
#     and planning latents are the same distribution, so the cached within_std
#     should match what the loss actually saw (~0.66), not diverge from it.
#
# NOTE ON ENV: deliberately run on the CURRENT env, pre-upstream-merge. The
# banked baselines were measured under this PushT block scale; merging the
# upstream scale fix (40->30) at the same time would confound two variables.
# The re-baseline under the corrected env is a separate, subsequent cell.
# ============================================================================

set -uo pipefail
cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate
export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs
ENC=lewm_pusht/weights.pt

run_eval () {  # $1=ckpt-relpath  $2=label  $3=steps
  local ckpt=$1 label=$2 st=$3
  [ -e "$STABLEWM_HOME/checkpoints/$ckpt" ] || { echo "[e6p7] SKIP $label"; return; }
  for es in 42 123; do
    echo "[e6p7] ==== EVAL $label @${st} (base pusht, n=50, eval seed $es) ===="
    local tmp="logs/_e6p7_${label}_${st}_${es}.log"
    if python scripts/plan/eval_wm.py --config-name pusht_diffusion \
        policy="$ckpt" cache_dir="$LOCAL_DATASET_DIR" seed=$es \
        "+model_overrides={num_inference_steps:$st}" > "$tmp" 2>&1; then
      grep -iE "success_rate" "$tmp" | tail -1
    else
      echo "[e6p7] EVAL FAILED — tail:"; tail -15 "$tmp"
    fi
  done
}

diagnose () {  # $1=name  $2=epoch  $3=source dataset  $4=lance prefix
  local name=$1 ep=$2 src=$3 pfx=$4
  echo "[e6p7] re-cache latents with the fine-tuned encoder (epoch $ep) ..."
  local LAT=${pfx}_${name}.lance
  python scripts/data/cache_latents.py \
    lewm_checkpoint=${name}/weights_epoch_${ep}.pt +from_dynamics=true \
    source=$src out_name=$LAT device=cuda batch_size=256 || return
  echo "[e6p7] latent geometry (frozen ref: within_std 0.829, mean_share 28%) ..."
  python scripts/data/latent_marginal_stats.py --dataset $LAT --n-episodes 50 \
    2>&1 | tail -3
  echo "[e6p7] Q2 multimodality (frozen 0.029; every e2e cell so far <=0.01) ..."
  python scripts/data/multimodality_diagnostic.py \
    --dataset $LAT --tag ${name}_latdyn \
    --mode dynamics --target-col latent --cond-cols latent action \
    --k 96 --n-anchors 4000 2>&1 | tail -3
}

case $SLURM_ARRAY_TASK_ID in

  0|1)
    # ---- warm-start arms: mirror e2e_warmstart_cell.sbatch exactly ----------
    SCALES=(0.1 0.03); IDX=$((SLURM_ARRAY_TASK_ID == 0 ? 1 : 0))
    LR_SCALE=${SCALES[$IDX]}
    SNAME=$(echo "ls$LR_SCALE" | tr '.' 'p')
    NAME="e6bn_x0_base_${SNAME}"
    SRC=${SRC:-dyn_x0_pusht_ps1}
    EP=2
    CKPT=$(ls -t "$STABLEWM_HOME/checkpoints/$SRC"/*.pt | head -1)
    [ -n "$CKPT" ] || { echo "no checkpoint under $SRC"; exit 1; }
    INIT="$SRC/$(basename "$CKPT")"
    echo "[e6p7] ===== $NAME | warm start $INIT | enc_lr_scale=$LR_SCALE | BN PINNED ====="

    python scripts/train/latent_diffusion.py \
      data=pusht data.dataset.name=pusht_expert_train.h5 \
      model.prediction_type=x0 model.num_inference_steps=20 \
      model.lewm_checkpoint=$ENC model.freeze_lewm=false \
      +model.frozen_encoder_norm=true \
      +init_dynamics_from="$INIT" \
      +encoder_lr_scale=$LR_SCALE +sigreg_weight=0.09 \
      output_model_name=$NAME seed=1 \
      trainer.max_epochs=$EP trainer.accelerator=gpu trainer.devices=1 \
      optimizer.lr=3e-4 loader.batch_size=32 num_workers=6 loader.num_workers=6 \
      checkpoint_interval=1 || { echo "[e6p7] TRAIN FAILED"; exit 1; }

    diagnose "$NAME" $EP pusht_expert_train.h5 pusht_base
    for ep in $(seq 1 $EP); do run_eval "${NAME}/weights_epoch_${ep}.pt" "${NAME}_ep${ep}" 1; done
    run_eval "${NAME}/weights_epoch_${EP}.pt" "${NAME}_ep${EP}" 20
    ;;

  2)
    # ---- from-scratch arm: mirror e2e_stopgrad_cell.sbatch (ARC 5c ps1) -----
    NAME="e6bn_diff_cslip05_ps1"
    EP=5
    echo "[e6p7] ===== $NAME | from-scratch e2e, v-pred, cslip05, BN PINNED ====="

    python scripts/train/latent_diffusion.py \
      data=pusht data.dataset.name=pusht_cslip05.lance \
      model.prediction_type=v model.num_inference_steps=20 \
      model.lewm_checkpoint=$ENC model.freeze_lewm=false \
      +model.stopgrad_target=true +model.frozen_encoder_norm=true \
      +encoder_lr_scale=0.1 +sigreg_weight=0.09 \
      output_model_name=$NAME seed=1 \
      trainer.max_epochs=$EP trainer.accelerator=gpu trainer.devices=1 \
      optimizer.lr=3e-4 loader.batch_size=128 num_workers=5 loader.num_workers=5 \
      checkpoint_interval=$EP || { echo "[e6p7] TRAIN FAILED"; exit 1; }

    diagnose "$NAME" $EP pusht_cslip05.lance pusht_cslip05
    run_eval "${NAME}/weights_epoch_${EP}.pt" "${NAME}" 1
    run_eval "${NAME}/weights_epoch_${EP}.pt" "${NAME}" 20
    ;;
esac

echo "DONE-E6P7-${SLURM_ARRAY_TASK_ID}"
