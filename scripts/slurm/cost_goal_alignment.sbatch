#!/bin/bash
#SBATCH -J swm-cost-align
#SBATCH -p gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# D3b probe (diagnosis 2026-07-23 / P2): is the latent goal-distance planning
# cost block-blind? Frozen vs e2e-fine-tuned encoder, cslip05 + base pusht.
set -uo pipefail
cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate
export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

run () {  # $1=dataset $2=encoder $3=tag
  echo "[align] ===== $3 ====="
  python scripts/data/cost_goal_alignment.py \
    --dataset "$1" --encoder "$2" --tag "$3" --n-episodes 40 || \
    echo "[align] $3 FAILED"
}

run pusht_cslip05.lance      lewm_pusht/weights.pt                     frozen_cslip05
run pusht_expert_train.lance lewm_pusht/weights.pt                     frozen_base
run pusht_cslip05.lance      e2c_diff_cslip05_ps1/weights_epoch_5.pt   e2c_ps1_cslip05
echo "DONE-COST-ALIGN"
