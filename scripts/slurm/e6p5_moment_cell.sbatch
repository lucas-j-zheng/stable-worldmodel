#!/bin/bash
#SBATCH -J ldm-e6p5-moment
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
# E6.5 stage B — MOMENT CONTROL: does pinning the latent location (and scale)
# during e2e fine-tuning recover the planning performance the warm start lost?
#
# Setup is E6.0a held fixed (same warm start from dyn_x0_pusht_ps1, same
# base-pusht pixels, same SIGReg 0.09, same stopgrad, same seed, encoder
# lr_scale 0.03 = the gentlest/best arm at 10/14 @1 vs frozen 20-22), with ONE
# thing added: an explicit moment penalty on the encoder's latent marginal.
# SIGReg constrains distribution SHAPE but demonstrably not LOCATION — both
# E6.0a arms drifted to mean_norm ~10 at batch 32.
#
#   arm 0  mean penalty, w=1    ||batch mean||^2 only
#   arm 1  mean + std, w=1      also pulls per-dim std -> 1 (tests the hidden
#                               per-dim shrink: median 0.6 vs frozen 0.97)
#   arm 2  mean penalty, w=10   dose check — if arm 0 fails to pin the mean,
#                               this says whether it was weight or mechanism
#
# PRE-REGISTERED READS (primary = base-pusht success @1, seeds 42/123, n=50)
#   (i)   mean_norm falls to ~0 AND @1 recovers toward 20  => the warm-start
#         deficit is latent location/scale drift; the diffusion-in-loop recipe
#         is FIXABLE and the e2e program continues on a moment-controlled
#         footing.
#   (ii)  mean_norm falls to ~0 but @1 stays ~10  => drift is a CORRELATE, not
#         the cause. Mechanism narrows to co-adaptation lag (denoiser chasing a
#         moving encoder). That is the pre-registered KILL for the "recipe-
#         fixable" reading => go to E6.4 (mixture/VQ head on frozen latents)
#         and the diffusion-proposer pivot.
#   (iii) mean_norm does NOT fall even at w=10  => the drift is being driven by
#         the dynamics loss itself, not by under-regularization; report as a
#         property of denoising-in-the-loop.
#   Secondary: latent residual_bimodal (frozen 0.029; every e2e cell so far
#   <=0.01). Centering is not expected to move it — a rise would be the first
#   Q2 movement in the program and would reopen E6.1.
#   Epoch-1 vs epoch-2 slope stays the churn readout (D3c).
# ============================================================================

set -uo pipefail
cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate
export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs "$STABLEWM_HOME/latstats"

NAMES=(e6m_mean_ls0p03 e6m_moms_ls0p03 e6m_mean10_ls0p03)
WEIGHTS=(1.0 1.0 10.0)
MATCH_STD=(false true false)

NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}
MW=${WEIGHTS[$SLURM_ARRAY_TASK_ID]}
MS=${MATCH_STD[$SLURM_ARRAY_TASK_ID]}
LR_SCALE=0.03
EP=2
ENC=lewm_pusht/weights.pt
SRC=${SRC:-dyn_x0_pusht_ps1}

CKPT=$(ls -t "$STABLEWM_HOME/checkpoints/$SRC"/*.pt | head -1)
[ -n "$CKPT" ] || { echo "no checkpoint under $SRC"; exit 1; }
INIT="$SRC/$(basename "$CKPT")"
echo "[e6m] ===== $NAME: warm start $INIT | moment_weight=$MW match_std=$MS | enc_lr_scale=$LR_SCALE ====="

python scripts/train/latent_diffusion.py \
  data=pusht data.dataset.name=pusht_expert_train.h5 \
  model.prediction_type=x0 model.num_inference_steps=20 \
  model.lewm_checkpoint=$ENC model.freeze_lewm=false \
  +init_dynamics_from="$INIT" \
  +encoder_lr_scale=$LR_SCALE +sigreg_weight=0.09 \
  +moment_weight=$MW +moment_match_std=$MS \
  output_model_name=$NAME seed=1 \
  trainer.max_epochs=$EP trainer.accelerator=gpu trainer.devices=1 \
  optimizer.lr=3e-4 loader.batch_size=32 num_workers=6 loader.num_workers=6 \
  checkpoint_interval=1 || { echo "[e6m] TRAIN FAILED"; exit 1; }

echo "[e6m] re-cache latents with the fine-tuned encoder (epoch $EP) ..."
LAT=pusht_base_${NAME}.lance
python scripts/data/cache_latents.py \
  lewm_checkpoint=${NAME}/weights_epoch_$EP.pt +from_dynamics=true \
  source=pusht_expert_train.h5 out_name=$LAT device=cuda batch_size=256

echo "[e6m] geometry: did the penalty pin the marginal? (E6.0a: mean_norm ~10, within_std ~0.6; frozen: ~0 / ~0.97)"
python scripts/data/latent_marginal_stats.py --dataset $LAT --n-episodes 50 \
  --dump-stats "$STABLEWM_HOME/latstats/${NAME}.npz" 2>&1 | tail -3

echo "[e6m] Q2 latent multimodality (frozen 0.029; every e2e cell so far <=0.01) ..."
python scripts/data/multimodality_diagnostic.py \
  --dataset $LAT --tag ${NAME}_latdyn \
  --mode dynamics --target-col latent --cond-cols latent action \
  --k 96 --n-anchors 4000 2>&1 | tail -3

# Q3: plain (unwhitened) cost, matching the E6.0a protocol exactly — whitening
# is stage A's variable and would confound the comparison if layered in here.
run_eval () {  # $1=epoch  $2=steps
  local ep=$1 st=$2
  for es in 42 123; do
    echo "[e6m] ==== EVAL $NAME epoch$ep @${st} (base, n=50, eval seed $es) ===="
    local tmp="logs/_e6m_${NAME}_ep${ep}_${st}_${es}.log"
    if python scripts/plan/eval_wm.py --config-name pusht_diffusion \
        policy=${NAME}/weights_epoch_$ep.pt cache_dir="$LOCAL_DATASET_DIR" \
        seed=$es "+model_overrides={num_inference_steps:$st}" > "$tmp" 2>&1; then
      grep -iE "success_rate" "$tmp" | tail -1
    else
      echo "[e6m] EVAL FAILED — tail:"; tail -20 "$tmp"
    fi
  done
}

# @1 first for BOTH epochs (primary readout + churn slope), then @20 secondary.
for ep in $(seq 1 $EP); do run_eval $ep 1; done
run_eval $EP 20

echo "DONE-E6M-$NAME"
