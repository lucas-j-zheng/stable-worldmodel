#!/bin/bash
#SBATCH -J ldm-slip-steps1
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

# Attribution probe for the multimodality-INDEPENDENT diffusion-dynamics
# advantage (slip 2x2: +6 bimodal, +10 unimodal — interaction falsified).
# Hypothesis: the gap is INFERENCE-TIME COMPUTE (20 DDIM refinement steps vs
# TMSE's single forward), not the objective. Test: same diffusion checkpoints,
# num_inference_steps=1 (v-param single-step x0 readout ~ one forward).
#   gap collapses to ~0 vs TMSE  -> iterative refinement is the advantage
#   gap survives at 1 step       -> the denoising OBJECTIVE trains a better
#                                    1-step predictor (training-time effect,
#                                    mirroring the policy-side variance story)
EP=100

for TAG in slip8 slip0; do
  SLIP=${TAG#slip}
  for s in 1 2 3; do
    name="dyn_diff_${TAG}_ps${s}"
    for es in 42 123; do
      echo "[steps1] ==== EVAL $name steps=1 (offset 25, n=50, eval seed $es) ===="
      python scripts/plan/eval_wm.py --config-name tworoom_slip_diffusion \
        policy=${name}/weights_epoch_$EP.pt cache_dir="$LOCAL_DATASET_DIR" \
        seed=$es world.slip_scale=$SLIP eval.dataset_name=tworoom_${TAG}.lance \
        "+model_overrides={num_inference_steps:1}" 2>&1 \
        | grep -iE "success_rate" | tail -1
    done
  done
done
echo "DONE-STEPS1-PROBE"
