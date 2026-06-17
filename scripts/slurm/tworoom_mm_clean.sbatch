#!/bin/bash
#SBATCH -J ldm-tworoom-mm2
#SBATCH -p gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate

export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

ENC=lewm_tworoom/weights_epoch_300.pt
COMMON="cache_dir=$STABLEWM_HOME num_traj=6000 world.num_envs=10 world.max_episode_steps=100"

# Collect NOISE-FREE pair on identical 2-door geometry: multimodal (random door)
# vs greedy (closest door). Apples-to-apples -- the only difference is door choice.
echo "[mm2] collect multimodal (noise-free) ..."
python scripts/data/collect_tworooms_multimodal.py $COMMON \
  +stochastic_door=true +out_name=tworoom_mm0.lance
echo "[mm2] collect greedy (noise-free) ..."
python scripts/data/collect_tworooms_multimodal.py $COMMON \
  +stochastic_door=false +out_name=tworoom_greedy0.lance

# Cache both through the same tworoom encoder.
for d in tworoom_mm0 tworoom_greedy0; do
  echo "[mm2] cache $d ..."
  python scripts/data/cache_latents.py lewm_checkpoint=$ENC \
    source=$d.lance out_name=${d}_latent.lance device=cuda batch_size=256
done

# Screen: latent-conditioned chunk (guarded) + state(agent)-conditioned single
# step. PREDICTION: multimodal residual_bimodal >> greedy now that noise is gone.
for d in tworoom_mm0 tworoom_greedy0; do
  echo "[mm2] screen $d latent-chunk ..."
  python scripts/data/multimodality_diagnostic.py \
    --dataset ${d}_latent.lance --tag ${d}_latentchunk \
    --mode policy_chunk --cond-from latent --chunk 8
  echo "[mm2] screen $d single-step policy ..."
  python scripts/data/multimodality_diagnostic.py \
    --dataset ${d}_latent.lance --tag ${d}_policy --mode policy
done

echo "[mm2] done. compare residual_bimodality_fraction: mm0 vs greedy0."
