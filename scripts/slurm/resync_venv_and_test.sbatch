#!/bin/bash
#SBATCH -J swm-resync-test
#SBATCH -p gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 00:45:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Re-sync the venv to the post-upstream-merge pyproject/uv.lock and run the
# test suite, to confirm the merged tree (solver -> planning/solver rename,
# new lance_video deps, bumped spt/datasets) works on Oscar.
set -uo pipefail
cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
mkdir -p logs
export MUJOCO_GL=egl

echo "=== HEAD ==="
git log --oneline -1

export UV=${UV:-$HOME/.local/bin/uv}
[ -x "$UV" ] || { echo "uv not found at $UV"; exit 1; }
echo "uv: $("$UV" --version)"

echo "=== uv sync (lockfile-driven, into project .venv) ==="
"$UV" sync
SYNC_RC=$?
echo "uv sync rc=$SYNC_RC"
[ $SYNC_RC -ne 0 ] && { echo "SYNC FAILED — aborting before tests"; exit $SYNC_RC; }

PY=.venv/bin/python
echo "=== import smoke test ==="
$PY -c "import stable_worldmodel as swm; from stable_worldmodel.planning import solver; from stable_worldmodel.planning.solver import diffusion_policy_solver; print('imports OK')" || { echo "IMPORT FAILED"; exit 1; }

echo "=== pytest ==="
$PY -m pytest -q 2>&1 | tail -60
echo "RESYNC-TEST-DONE rc=${PIPESTATUS[0]}"
