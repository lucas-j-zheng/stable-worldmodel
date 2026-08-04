#!/bin/bash
#SBATCH -J swm-e6p6-blind
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -x gpu3005
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# ============================================================================
# E6.6 — AGENT-BLINDNESS: the last mechanism standing for the warm-start deficit
#
# Why the field narrowed to this. 4541921 killed the mean-drift story outright:
# the FROZEN encoder's cache measures mean_norm 7.24 — and it plans at 20-22.
# A warm-start mean of 9.87 therefore cannot be what halves planning, and the
# "frozen ~0" reference that motivated E6.5 was never a measurement (it was the
# SIGReg prior, hard-coded into the diagnostic's reference string). Whitening is
# dead too, and for a sharper reason than a null: it DROPPED the frozen baseline
# to 10/16, so it is an actively harmful intervention, not a no-op.
#
# What survives is A2: fine-tuning drove the planning cost's sensitivity to
# AGENT position to ~zero (standardized coef 0.124 frozen -> 0.013 / -0.052
# warm; Spearman 0.006 / 0.066), while block sensitivity held at ~0.50. If the
# goal-distance cost cannot see the pusher, the planner gets no ranking signal
# for the approach phase — only block motion scores, and most candidate action
# sequences move no block. That degrades planning without flooring it, which is
# the shape actually observed (~2x).
#
# Two independent tests, because A2 alone rests on an index assumption:
#
#   M1  MEASUREMENT, label-free. The probe assumed agent=dims 0,1 / block=2,3,
#       but this dataset reports state_dim=7. New --per-dim output regresses the
#       latent goal-distance on EVERY state dim separately, so "the cost stopped
#       tracking dim j" is readable with no labeling at all. Frozen vs both warm
#       arms vs a moment arm, same dataset, same protocol.
#   M2  BEHAVIOR, no new GPU work. eval_wm already wrote per-step
#       positions_off25_seed{42,123}.npz into every checkpoint dir. If the cost
#       is agent-blind, warm-start failures should concentrate in the APPROACH
#       phase (agent never reaches the block) rather than the push phase (agent
#       contacts, block goes wrong). failure_positions.py compares the frozen
#       baseline against the warm arms directly.
#
# PRE-REGISTERED READS
#   M1 reproduces a near-zero coefficient on the SAME dims across both warm arms
#      while frozen is clearly nonzero  =>  agent-blindness confirmed as a real,
#      labeling-independent property of the fine-tuned encoder.
#   M1 shows the collapse is spread over all dims (everything shrinks)  =>  not
#      agent-specific; it is a general loss of state sensitivity, and the story
#      becomes "fine-tuning degrades the cost's state coverage", weaker but
#      still publishable.
#   M2 warm failures shift toward approach-phase  =>  behavioral corroboration;
#      mechanism is then supported by measurement AND outcome, and the fix is a
#      block-pose/agent-pose auxiliary loss (E6.6b) or an agent-aware cost.
#   M2 shows no phase shift  =>  the measurement is real but behaviorally inert,
#      same class of outcome as the P0a smoothing test (candidate exists at the
#      regression level, explains nothing closed-loop). That would leave the
#      warm-start deficit with NO surviving mechanism => stop mechanism-hunting
#      and go to E6.4 / the proposer pivot.
# ============================================================================

set -uo pipefail
cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate
export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

# ------------------------------------------------- M1: label-free per-dim sweep
echo "[e6p6] ===== M1: per-state-dim cost alignment (frozen vs fine-tuned) ====="
for pair in "lewm_pusht/weights.pt:frozen_base_perdim" \
            "e2w_x0_base_ls0p03/weights_epoch_2.pt:warm_ls0p03_perdim" \
            "e2w_x0_base_ls0p1/weights_epoch_2.pt:warm_ls0p1_perdim" \
            "e6m_mean_ls0p03/weights_epoch_2.pt:moment_w1_perdim"; do
  ckpt=${pair%%:*}; tag=${pair##*:}
  echo "[e6p6] --- $tag ---"
  python scripts/data/cost_goal_alignment.py \
    --dataset pusht_expert_train.h5 --encoder "$ckpt" \
    --tag "$tag" --n-episodes 40 || echo "[e6p6] M1 $tag FAILED"
done

# --------------------------------------- M2: failure-phase analysis (no new GPU)
echo "[e6p6] ===== M2: failure positions, frozen baseline vs warm arms ====="
python scripts/data/failure_positions.py \
  --ckpt-root "$STABLEWM_HOME/checkpoints" \
  --models dyn_x0_pusht_ps1 e2w_x0_base_ls0p03 e2w_x0_base_ls0p1 \
           e6m_mean_ls0p03 e6m_moms_ls0p03 \
  --out logs/e6p6_failure_positions.json || echo "[e6p6] M2 FAILED"

# ------------------------------- path probe, repaired (pipeline hygiene, not a
# mechanism test any more: frozen's cached mean is 7.24, so the question is now
# why a SIGReg-pretrained encoder caches off-center at all — it affects every
# cached-latent number in the program, including the bimodality screens.
echo "[e6p6] ===== path attribution (repaired THWC->TCHW layout) ====="
for pair in "lewm_pusht/weights.pt:frozen" \
            "e2w_x0_base_ls0p03/weights_epoch_2.pt:warm_ls0p03"; do
  ckpt=${pair%%:*}; tag=${pair##*:}
  python scripts/data/latent_path_probe.py \
    --encoder "$ckpt" --tag "$tag" --n-episodes 10 \
    || echo "[e6p6] path probe $tag FAILED"
done

echo "DONE-E6P6"
