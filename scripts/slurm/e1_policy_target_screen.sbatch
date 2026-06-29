#!/bin/bash
#SBATCH -J ldm-e1-screen
#SBATCH -p gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# E1, step 1: the TRUE multimodality dose of the door_prob datasets, measured
# with policy_target (conditions on the episode's FINAL destination = door-
# agnostic, does NOT leak the route). The 2026-06-26 measure-dose used
# policy_goal, which leaks the door by conditioning on near-future state -> read
# flat. If policy_target shows a clean monotone dose 0.5->1.0, door_prob is a
# REAL behavioral knob (the earlier "leaky" verdict was a screen artifact) and we
# already have a measured x-axis on existing, in-distribution (2-door) data to
# plot the DP-vs-MLP gap against. Encoder-free, ~2 min.

set -euo pipefail

cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate

export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

# door_prob: mm0=0.5 (max bimodal) .. p10=1.0 (always door 0, unimodal).
# greedy0 = 1-door greedy = truly-unimodal low anchor (different geometry).
for ds in tworoom_mm0 tworoom_p07 tworoom_p09 tworoom_p10 tworoom_greedy0; do
  echo "==== policy_target screen: ${ds} ===="
  python scripts/data/multimodality_diagnostic.py \
    --dataset "${ds}.lance" --tag "e1_${ds}" \
    --mode policy_target --chunk 8 --early-frac 0.5 \
    --k 96 --n-anchors 4000 || echo "[e1] ${ds} FAILED (missing dataset?)"
done

echo "[e1] DONE. residual_bimodality_fraction by door_prob:"
echo "      mm0(0.5) p07(0.7) p09(0.9) p10(1.0) greedy0(1-door) -> logs/multimodality_e1_*.json"
