#!/bin/bash
#SBATCH -J hjepa-3door
#SBATCH -p gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# H-JEPA E14 + E15:
#
# E14 3-DOOR TRIMODAL FALSIFICATION TEST of the mean-in-gap mechanism.
#   Collect TwoRoom with doors at {40, 112(center), 184}, expert commits
#   uniformly among the 3. By symmetry the conditional mean at route-ambiguous
#   anchors lies near the CENTER route -- a real mode.
#   PRE-REGISTERED (HJEPA_REASONING.md E14):
#     1. Screen (goal-conditioned, K=8/none): multimodality >= 2-door levels
#        (gmm2_frac >= 0.6) BUT gap_ratio median SUBSTANTIALLY higher than the
#        2-door 0.49 (predict > 0.7): a mode sits at/near the mean.
#     2. Bench: MSE precision RECOVERS toward the p10 level (predict >= 0.75 vs
#        0.61 on 2-door mm0) despite the multimodality -- multimodality per se
#        does not hurt MSE; mean-in-gap does.
#     3. MSE mode_coverage stays LOW (~1/3: it covers only the center route) --
#        precision and coverage DISSOCIATE: feasible but no menu.
#   FALSIFIED IF: MSE precision on 3-door stays ~0.6 (then the deficit is about
#   multimodality generally, and gap_ratio is not the operative mechanism).
#
# E15 LATENT BENCH RETRY: latent-cached datasets have no goal column ->
#   --goal-from final (destination conditioning, validated by policy_target).
#   Pre-registration unchanged from E10.

set -uo pipefail

cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate

export MUJOCO_GL=egl
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

DS3=tworoom_3door.lance

# E14.1 collect 3-door (uniform commitment among 3 fitting doors)
if [ ! -d "$STABLEWM_HOME/datasets/${DS3}" ] && [ ! -d "$STABLEWM_HOME/${DS3}" ]; then
  python scripts/data/collect_tworooms_multimodal.py \
    cache_dir="$STABLEWM_HOME" +n_doors=3 +door_prob=0.5 +out_name=$DS3
fi

# E14.2 goal-conditioned screen at K in {1,8}
for K in 1 8; do
  for COARSE in none sum; do
    python scripts/data/multimodality_diagnostic.py \
      --dataset $DS3 --tag "hjepa_3door_k${K}_${COARSE}" \
      --mode dynamics_k --horizon $K --coarse-action $COARSE \
      --target-col state --cond-cols state goal_state \
      --k 96 --n-anchors 4000 \
      || echo "[3door] SKIP screen K=$K $COARSE"
  done
done

# E14.3 proposer bench on 3-door
python scripts/train/proposer_bench.py \
  --dataset $DS3 --horizon 8 --epochs 25 \
  --n-anchors 1000 --k-emp 64 --m-samples 32 \
  || echo "[3door] SKIP bench"

# E15 latent bench retry (destination goal)
for DS in tworoom_mm0_latent tworoom_p10_latent; do
  python scripts/train/proposer_bench.py \
    --dataset ${DS}.lance --state-col latent --goal-from final \
    --horizon 8 --epochs 25 --n-anchors 800 --k-emp 64 --m-samples 32 \
    || echo "[3door] SKIP latent $DS"
done

echo "[3door] DONE."
