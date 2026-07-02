#!/bin/bash
#SBATCH -J ldm-var-probe
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

# Variance-mechanism probe for the P0a keeper finding (DP cross-seed sd 3.6 vs
# TMSE 7.3, same backbone/data): is the stabilization a SAMPLING effect or an
# OBJECTIVE/training effect? Re-eval the 8 seeded mm05 DP checkpoints under
# sampling-knob variations (no retraining):
#   base    : eta=0,  steps=20 (training default — reproduces the 3.6)
#   eta1    : eta=1.0,steps=20 (ancestral noise — if sd jumps, sampling matters)
#   steps5  : eta=0,  steps=5  (coarser DDIM — path-averaging reduced)
#   steps50 : eta=0,  steps=50 (finer DDIM)
# Read: cross-seed sd per config. Flat sd across configs => the objective/loss
# landscape is the stabilizer (training-time), not the sampler.
EP=250

run_cfg () {  # $1=tag $2=overrides
  echo "[varprobe] ===== config $1 ====="
  for s in 1 2 3 4 5 6 7 8; do
    ck="dp_mm05_ps${s}/weights_epoch_$EP.pt"
    echo "[varprobe] EVAL $1 seed $s"
    python scripts/plan/eval_wm.py --config-name diffusion_policy_eval_mm \
      policy=$ck cache_dir="$LOCAL_DATASET_DIR" \
      seed=42 eval.goal_offset_steps=8 eval.num_eval=50 $2 2>&1 \
      | grep -iE "success_rate" | tail -1
  done
}

run_cfg base    ""
run_cfg eta1    "+model_overrides={eta:1.0}"
run_cfg steps5  "+model_overrides={num_inference_steps:5}"
run_cfg steps50 "+model_overrides={num_inference_steps:50}"
echo "DONE-VAR-PROBE"
