#!/bin/bash
#SBATCH -J ldm-e2e-evaldbg
#SBATCH -p gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -uo pipefail
cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate
export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

# ARC 5 eval-crash repro: the cell sbatch piped eval through `grep success_rate`
# which swallowed the traceback (pipefail then killed the job after the screen).
# Run ONE eval on an e2e checkpoint with the full output visible.
python scripts/plan/eval_wm.py --config-name pusht_diffusion \
  policy=e2e_diff_cslip05_ps1/weights_epoch_5.pt cache_dir="$LOCAL_DATASET_DIR" \
  seed=42 world.contact_slip_scale=0.5 eval.dataset_name=pusht_cslip05.lance \
  "+model_overrides={num_inference_steps:1}" 2>&1 | tail -60
echo "DEBUG-EVAL-DONE rc=$?"
