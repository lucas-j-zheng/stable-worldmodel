#!/bin/bash
#SBATCH -J swm-warm-eval
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 06:00:00
#SBATCH -x gpu3005
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# E6.0a warm-start Q3 verdict, split out of the full replay (which paces
# ~12min/eval and risks walltime before reaching these). @1 evals FIRST —
# they are the primary readout (universal sampling penalty) and the verdict.
set -uo pipefail
cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate
export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

run_eval () {  # $1=ckpt $2=label $3=steps
  local ckpt=$1 label=$2 st=$3
  [ -e "$STABLEWM_HOME/checkpoints/$ckpt" ] || { echo "[warm-eval] SKIP $label"; return; }
  for es in 42 123; do
    echo "[warm-eval] ==== EVAL $label @${st} steps (n=50, eval seed $es) ===="
    local tmp="logs/_warmeval_${label}_${st}_${es}.log"
    if python scripts/plan/eval_wm.py --config-name pusht_diffusion \
        policy=$ckpt cache_dir="$LOCAL_DATASET_DIR" seed=$es \
        "+model_overrides={num_inference_steps:$st}" > "$tmp" 2>&1; then
      grep -iE "success_rate" "$tmp" | tail -1
    else
      echo "[warm-eval] EVAL FAILED — tail:"; tail -12 "$tmp"
    fi
  done
}

# @1 pass first (the verdict), then @20
for ST in 1 20; do
  for LS in ls0p1 ls0p03; do
    for EPN in 1 2; do
      run_eval "e2w_x0_base_${LS}/weights_epoch_${EPN}.pt" "warm_${LS}_ep${EPN}" $ST
    done
  done
done
echo "DONE-WARM-EVAL"
