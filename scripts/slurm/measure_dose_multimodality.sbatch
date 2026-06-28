#!/bin/bash
#SBATCH -J ldm-measure-dose
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

# Measure the ACTUAL policy multimodality of each dose dataset (the true x-axis
# the nominal door_prob was a proxy for). policy_goal: cond=(state_t, state_t+8)
# = (current, synthesized goal exactly as eval forms it), target = 8-step action
# chunk. residual_bimodality_fraction = the literal dose of P(action|state,goal)
# multimodality. Pair vs the DP-MLP gaps to get the MEASURED dose-response.
# Encoder-free (uses state+action copied into the latent cache).
for DS in tworoom_mm_latent2:mm05 tworoom_p07_latent:p07 tworoom_p09_latent:p09 \
          tworoom_p10_latent:p10 tworoom_greedy2_latent:greedy2; do
  name="${DS%%:*}"; tag="${DS##*:}"
  echo "==== $tag ($name) ===="
  python scripts/data/multimodality_diagnostic.py \
    --dataset ${name}.lance --tag dose_${tag} \
    --mode policy_goal --goal-offset 8 --chunk 8 2>&1 \
    | grep -iE "RESULT|det_r2|residual_bimodality_fraction|pairs|absent|Error" | tail -8 \
    || echo "(dataset $name not ready / failed)"
done
echo "DONE-MEASURE-DOSE"
