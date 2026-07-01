#!/bin/bash
#SBATCH -J hjepa-pusht-kstep
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# H-JEPA RUNG 1.5a (plan review C6): does PushT branch at K-step?
# PushT read near-deterministic/unimodal at 1-step (det_R2 0.86, 06-16 screen).
# Contact manipulation COULD branch at K-step (approach the T from the left vs
# right = a contact-mode route choice) -- if so, a manipulation domain for the
# paper at zero collection cost.
#
# PRE-REGISTERED PREDICTIONS:
#   - scripted expert (pusht_latent): stays ~flat at all K -- single scripted
#     route, no demonstrated branching (behavioral-reachability caveat B7).
#   - multi-human (pusht_human_*): the live cell. If bimodal_frac rises with K
#     under sum/none, PushT is rescued at K-step exactly like the doors domain.
#   - EITHER outcome is informative: flat scripted + rising human confirms
#     "coverage requirement" (modes must be demonstrated); flat both = PushT
#     lacks K-scale route structure under these demos.
#
# Cells guarded (|| echo SKIP): dataset column availability differs across the
# cached pusht variants; the diagnostic exits with the column list on mismatch.

set -uo pipefail

cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate

export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

for DS in pusht_latent pusht_human_state; do
  for K in 1 4 8; do
    for COARSE in full sum none; do
      echo "[hjepa-pusht] $DS K=$K coarse=$COARSE ..."
      python scripts/data/multimodality_diagnostic.py \
        --dataset ${DS}.lance --tag "hjepa_pusht_${DS}_k${K}_${COARSE}" \
        --mode dynamics_k --horizon $K --coarse-action $COARSE \
        --target-col state --cond-cols state \
        --k 96 --n-anchors 4000 \
        || echo "[hjepa-pusht] SKIP $DS K=$K $COARSE (see error above)"
    done
  done
done

# latent-target arm on the human data (does branching survive the encoder,
# manipulation edition)
for K in 1 4 8; do
  for COARSE in full sum none; do
    echo "[hjepa-pusht] pusht_human_latent K=$K coarse=$COARSE (latent target) ..."
    python scripts/data/multimodality_diagnostic.py \
      --dataset pusht_human_latent.lance \
      --tag "hjepa_pusht_humanlat_k${K}_${COARSE}" \
      --mode dynamics_k --horizon $K --coarse-action $COARSE \
      --target-col latent \
      --k 96 --n-anchors 4000 \
      || echo "[hjepa-pusht] SKIP humanlat K=$K $COARSE (see error above)"
  done
done

echo "[hjepa-pusht] DONE. JSONs: logs/multimodality_hjepa_pusht_*.json"
