#!/bin/bash
#SBATCH -J ldm-e2e-latstats
#SBATCH -p batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -uo pipefail
cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate
export STABLEWM_HOME=${STABLEWM_HOME:-/oscar/scratch/$USER/stable_worldmodel}
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-$STABLEWM_HOME}
mkdir -p logs

# ARC 5 diagnostic: latent-marginal drift of e2e encoders vs the frozen SIGReg
# reference. If e2e latents' scale/mean drifted off (0, 1), that explains the
# uniform closed-loop floor (planner calibration broke), not lost dynamics.
for DS in pusht_cslip05_latent.lance \
          pusht_cslip05_e2e_diff_cslip05_ps1.lance \
          pusht_cslip05_e2e_diff_cslip05_ps2.lance \
          pusht_cslip05_e2e_diff_cslip05_ps3.lance \
          pusht_cslip05_e2e_tmse_cslip05_ps1.lance \
          pusht_cslip05_e2e_tmse_cslip05_ps2.lance \
          pusht_cslip05_e2e_tmse_cslip05_ps3.lance; do
  python scripts/data/latent_marginal_stats.py --dataset $DS --n-episodes 50 2>&1 | tail -4
done
echo "DONE-E2E-LATSTATS"
