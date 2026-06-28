#!/bin/bash
#SBATCH -J ldm-dose-target
#SBATCH -p gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 00:40:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate
export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

# CORRECTED dose measurement: condition on (state_t, episode-FINAL state = the
# destination, same for both doors -> no route leak), early half of each episode.
# This is the real P(action|state,target) multimodality the door knob should
# control. policy_goal (realized near-future goal) leaked the door -> flat ~0.14;
# this should reveal random-door >> greedy/door_prob=1.0 if the dose is real.
for DS in tworoom_mm_latent2:mm05 tworoom_p07_latent:p07 tworoom_p09_latent:p09 \
          tworoom_p10_latent:p10 tworoom_greedy2_latent:greedy2; do
  name="${DS%%:*}"; tag="${DS##*:}"
  echo "==== $tag ($name) ===="
  python scripts/data/multimodality_diagnostic.py \
    --dataset ${name}.lance --tag target_${tag} \
    --mode policy_target --chunk 8 --early-frac 0.5 2>&1 \
    | grep -iE "RESULT|det_r2_mean|residual_bimodality_fraction|pairs,|absent|Error" | tail -8 \
    || echo "(dataset $name not ready / failed)"
done
echo "DONE-DOSE-TARGET"
