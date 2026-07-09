#!/bin/bash
#SBATCH -J ldm-slip16-prep
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate
export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

# ARC 3 — DOES EXTREME STOCHASTICITY BREAK THE SAMPLING PENALTY?
# The one universal finding of the whole program is that a one-shot latent
# readout (@1 step) matches or beats 20-step DDIM sampling for planning, in
# every cell of both domains. But all cells had either near-deterministic
# dynamics OR modest slip (S<=8), AND receding-horizon replanning (every 5
# steps) that re-observes the realized coin before it compounds. If sampling
# EVER pays, it pays where there is the most distribution to integrate over:
# extreme per-step aleatoric slip. This is the pre-registered falsification
# test. slip16 = 2x the strongest dose previously trained (slip8).
#
# Stage 1 (this job): collect the slip16 dataset, cache latents through the
# EXISTING frozen lewm_tworoom encoder (slip changes dynamics, not appearance
# -> encoder in-distribution, no retrain), and confirm the LATENT-space
# bimodality the dynamics model will actually condition on. Gate the chain on
# it (must clear 0.3; expect ~saturated near 0.98).
S=16
DS=tworoom_slip${S}.lance
LAT=tworoom_slip${S}_latent.lance

echo "[slip16] collecting $DS (slip_scale=$S) ..."
python scripts/data/collect_tworooms_slip.py \
  cache_dir="$STABLEWM_HOME" num_traj=4000 \
  world.num_envs=10 world.max_episode_steps=100 \
  +slip_scale=$S +action_noise=2.0 +out_name=$DS

echo "[slip16] caching latents -> $LAT ..."
python scripts/data/cache_latents.py \
  lewm_checkpoint=lewm_tworoom/weights_epoch_300.pt \
  source=$DS out_name=$LAT device=cuda batch_size=256

echo "[slip16] LATENT dynamics screen slip_scale=$S ..."
python scripts/data/multimodality_diagnostic.py \
  --dataset $LAT --tag slip${S}_latdyn \
  --mode dynamics --target-col latent --cond-cols latent action \
  --k 96 --n-anchors 4000

python - <<'EOF'
import json
d = json.load(open('logs/multimodality_slip16_latdyn.json'))
frac = d.get('residual_bimodality_fraction', 0.0)
print(f'[slip16] GATE latent residual_bimodal = {frac:.3f} '
      f'({"PASS" if frac > 0.3 else "FAIL"})')
EOF
echo "DONE-SLIP16-PREP"
