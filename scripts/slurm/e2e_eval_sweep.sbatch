#!/bin/bash
#SBATCH -J ldm-e2e-evalsweep
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 05:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -uo pipefail
cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate
export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

# ARC 5 stage 2 — EVAL-ONLY sweep over the saved e2e checkpoints (training and
# screens completed; the cell's evals died on a hydra struct error:
# world.contact_slip_scale needed the +append prefix in pusht_diffusion.yaml).
# THE P3 QUESTION: the e2e-diffusion latent now CARRIES the branching
# (bimodality 0.11 vs frozen 0.029). Does 20-step sampling finally beat the
# one-shot readout in a multimodality-preserving latent space? If yes — the
# program's first genuine sampling win. Also: e2e-diff vs e2e-tmse closed-loop
# (does latent-shaping pay in control?).
# Tracebacks preserved: each eval logs to a file; failures print the tail
# instead of being swallowed by grep+pipefail (the bug that killed the cells).
TAG=cslip05
SLIP=0.5
EP=5

run_eval () {  # $1=name $2=steps(""=default tmse)
  local name=$1 st=${2:-}
  local extra=""
  [ -n "$st" ] && extra="+model_overrides={num_inference_steps:$st}"
  for es in 42 123; do
    echo "[e2e-sweep] ==== EVAL $name steps=${st:-det} (n=50, eval seed $es) ===="
    local tmp="logs/_ev_${name}_${st:-det}_${es}.log"
    if python scripts/plan/eval_wm.py --config-name pusht_diffusion \
        policy=${name}/weights_epoch_$EP.pt cache_dir="$LOCAL_DATASET_DIR" \
        seed=$es +world.contact_slip_scale=$SLIP \
        eval.dataset_name=pusht_${TAG}.lance $extra > "$tmp" 2>&1; then
      grep -iE "success_rate" "$tmp" | tail -1
    else
      echo "[e2e-sweep] EVAL FAILED — traceback tail:"
      tail -25 "$tmp"
    fi
  done
}

if [ "${SKIP_DIFF:-0}" != 1 ]; then
  for s in 1 2 3; do
    run_eval "e2e_diff_${TAG}_ps${s}" 1
    run_eval "e2e_diff_${TAG}_ps${s}" 20
  done
fi
for s in 1 2 3; do
  # tmse checkpoints ps2/ps3 exist once jobs 3830136/37 finish training
  if [ -e "$STABLEWM_HOME/checkpoints/e2e_tmse_${TAG}_ps${s}/weights_epoch_$EP.pt" ] \
     || [ -e "$(python -c 'import stable_worldmodel as swm; print(swm.data.utils.get_cache_dir(sub_folder="checkpoints"))' 2>/dev/null)/e2e_tmse_${TAG}_ps${s}/weights_epoch_$EP.pt" ]; then
    run_eval "e2e_tmse_${TAG}_ps${s}" ""
  else
    echo "[e2e-sweep] SKIP e2e_tmse_${TAG}_ps${s} (checkpoint not found yet)"
  fi
done
echo "DONE-E2E-EVAL-SWEEP"
