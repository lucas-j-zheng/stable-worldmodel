#!/bin/bash
#SBATCH -J ldm-slip16-cadence
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate
export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

# ARC 3b — the mechanistic pin (EVAL-ONLY, reuses the 6 slip16 checkpoints).
# ARC 3 CONFIRMED the sampling penalty survives 0.974-bimodal slip16 (diff@1 79.3
# > @5 77.0 > @20 74.3, monotonic). Hypothesis for WHY: receding-horizon
# replanning (rh=5) re-observes the realized slip coin before it compounds, so the
# conditional MEAN is sufficient and sampling the distribution buys nothing.
# TEST: re-eval at receding_horizon=8 (= execute the full 8-step plan before
# replanning; minimal correction) vs the rh=5 baseline already measured.
#   If diff@20 catches/passes diff@1 at rh=8 -> replanning WAS what forgave the
#     stochasticity; reversibility (not multimodality) is the operative variable.
#   If diff@1 still dominates at rh=8 -> the penalty is deeper than replanning.
# steps {1,20} for diff (the trend is monotone; skip 5), @1 for TMSE.
SLIP=16
TAG=slip16
EP=${EP:-100}
RH=${RH:-8}

for s in 1 2 3; do
  name="dyn_diff_${TAG}_ps${s}"
  for st in 1 20; do
    for es in 42 123; do
      echo "[slip16-cadence] ==== EVAL $name steps=$st rh=$RH (n=50, eval seed $es) ===="
      python scripts/plan/eval_wm.py --config-name tworoom_slip_diffusion \
        policy=${name}/weights_epoch_$EP.pt cache_dir="$LOCAL_DATASET_DIR" \
        seed=$es world.slip_scale=$SLIP eval.dataset_name=tworoom_${TAG}.lance \
        plan_config.receding_horizon=$RH \
        "+model_overrides={num_inference_steps:$st}" 2>&1 \
        | grep -iE "success_rate" | tail -1
    done
  done
  tname="dyn_tmse_${TAG}_ps${s}"
  for es in 42 123; do
    echo "[slip16-cadence] ==== EVAL $tname (TMSE) rh=$RH (n=50, eval seed $es) ===="
    python scripts/plan/eval_wm.py --config-name tworoom_slip_diffusion \
      policy=${tname}/weights_epoch_$EP.pt cache_dir="$LOCAL_DATASET_DIR" \
      seed=$es world.slip_scale=$SLIP eval.dataset_name=tworoom_${TAG}.lance \
      plan_config.receding_horizon=$RH 2>&1 | grep -iE "success_rate" | tail -1
  done
done
echo "DONE-SLIP16-CADENCE-rh$RH"
