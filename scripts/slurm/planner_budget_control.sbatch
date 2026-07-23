#!/bin/bash
#SBATCH -J swm-planbudget
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# PLANNER-BUDGET CONTROL (2026-07-23). The LeWM paper (arXiv 2603.19312) plans
# PushT with CEM 300 samples / 30 elites (10%) / 30 iters -> ~80% success. The
# program's diffusion eval overrides num_samples=64 with topk=30 (47% elite
# fraction, near-zero selection pressure) -> frozen MSE 30.7, x0 20-22. Same
# encoder, same latent space. If paper-budget CEM lifts the frozen baselines
# substantially, the program's absolute success scale was planner-compressed
# and every "floor" needs re-reading at the wider detection range.
# PRE-REGISTERED: tmse >= ~50 at 300 samples => budget-compression confirmed;
# unchanged (~30) => budget not binding, paper gap lies elsewhere (predictor
# arch/training or horizon-5-vs-8).
set -uo pipefail
cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate
export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

run_eval () {  # $1=policy $2=label $3=extra overrides
  local pol=$1 label=$2 extra=${3:-}
  for es in 42 123; do
    echo "[budget] ==== EVAL $label num_samples=300 (n=50, eval seed $es) ===="
    local tmp="logs/_budget_${label}_${es}.log"
    if python scripts/plan/eval_wm.py --config-name pusht_diffusion \
        policy=$pol cache_dir="$LOCAL_DATASET_DIR" \
        seed=$es solver.num_samples=300 $extra > "$tmp" 2>&1; then
      grep -iE "success_rate" "$tmp" | tail -1
    else
      echo "[budget] EVAL FAILED — traceback tail:"; tail -15 "$tmp"
    fi
  done
}

# newest .pt in each frozen run
pick () { ls -t "$STABLEWM_HOME/checkpoints/$1"/*.pt | head -1 | xargs basename; }

TM=$(pick dyn_tmse_pusht_ps1)
X0=$(pick dyn_x0_pusht_ps1)
echo "[budget] tmse ckpt: $TM   x0 ckpt: $X0"

run_eval dyn_tmse_pusht_ps1/$TM  tmse300   ""
run_eval dyn_x0_pusht_ps1/$X0    x0_300_s1 "+model_overrides={num_inference_steps:1}"
echo "DONE-PLANBUDGET"
