#!/bin/bash
#SBATCH -J hjepa-kstep-screen
#SBATCH -p gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# H-JEPA RUNG 1 GATE: is the K-step-ahead conditional p(z_{t+K} | z_t, coarse_a)
# MULTIMODAL once the fine action sequence a_{t:t+K} is (partly) marginalized out?
# Encoder-free (next-STATE) screen -- the cheapest possible pre-build gate, before
# any level-2 encoder g or diffusion training. Mirrors the POMDP dynamics screen.
#
#   Two axes:  horizon K in {1,2,4,8}  x  coarse-action in {full, sum, none}.
#   THE SIGNATURE THAT GREENLIGHTS H-JEPA:
#     - coarse=full  : residual_bimodal_frac stays LOW for all K (composed
#                      determinism -- all fine control kept, no branching). CONTROL.
#     - coarse=sum/none : residual_bimodal_frac RISES with K (controllable
#                      reachability -- the diffusion edge for a level-2 predictor).
#   QUANTITATIVE PREDICTION (door_prob=0.5, episode len ~T): bimodal_frac(K) ~ 0.5*K/T.
#     For T~40, K=8 -> ~0.10 on coarse=sum. Hold the K=8/sum cell against ~0.10.
#   FAIL = flat & low under ALL conditioning -> reachable sets are smooth, not
#     branching; the domain lacks hard route structure. Kill before building g.
#
# Read residual_bimodality_fraction (SHAPE); the modes here are hard-separated
# (two doors, two rooms) so Sarle BC has power -- unlike the soft policy case.

set -euo pipefail

cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate

export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

# 0. Instrument sanity: still flags synthetic bimodal, clears unimodal.
echo "[hjepa] self-test (expect unimodal~0, bi/tri HIGH) ..."
python scripts/data/multimodality_diagnostic.py --self-test

# 1. Collect a MULTIMODAL TwoRoom set: 2-door geometry, expert commits to a random
#    door per episode (door_prob=0.5 = maximal route ambiguity). state + action
#    columns are all the encoder-free screen needs.
DS=tworoom_mm_dp05.lance
echo "[hjepa] collecting $DS (door_prob=0.5) ..."
python scripts/data/collect_tworooms_multimodal.py \
  cache_dir="$STABLEWM_HOME" +door_prob=0.5 +out_name=$DS

# 2. K x coarse grid, encoder-free next-state target. det_R2 is expected to DROP
#    with K for coarse=sum/none (reachable set widens); the diagnostic is the
#    residual_bimodal_frac (does that widening BRANCH?).
for K in 1 2 4 8; do
  for COARSE in full sum none; do
    echo "[hjepa] screen K=$K coarse=$COARSE ..."
    python scripts/data/multimodality_diagnostic.py \
      --dataset $DS --tag "hjepa_k${K}_${COARSE}" \
      --mode dynamics_k --horizon $K --coarse-action $COARSE \
      --target-col state --cond-cols state \
      --k 96 --n-anchors 4000
  done
done

echo "[hjepa] DONE. Build the K x coarse table from:"
echo "        logs/multimodality_hjepa_k{1,2,4,8}_{full,sum,none}.json"
echo "        -> field residual_bimodality_fraction."
echo "  GREENLIGHT: full stays ~flat/low; sum & none RISE with K; k8/sum ~ 0.10."
