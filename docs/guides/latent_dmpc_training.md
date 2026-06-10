---
title: Latent D-MPC Training
summary: Local and Oscar setup for training latent diffusion dynamics from LeWM latents
---

This guide is the minimum path to start the Latent D-MPC experiment:

1. Get a TwoRoom dataset into the stable-worldmodel cache.
2. Get or train a pretrained LeWM checkpoint.
3. Cache frozen LeWM latents for the dataset.
4. Train the latent diffusion dynamics model.
5. Run Stage 0 open-loop evaluation before trying MPC.
6. Run TwoRoom D-MPC only after Stage 0 is sane.

The default commands below target `swm/TwoRoom-v1`. Do real training on Oscar or
another Linux machine with an NVIDIA GPU. Local CPU runs are useful for smoke
tests only.

## What the experiment is

The MVP is a latent-space dynamics model:

- LeWM encodes pixels into SIGReg latents `z`.
- A diffusion model predicts future latent trajectories conditioned on latent
  history and actions.
- Stage 0 compares open-loop latent prediction against LeWM's deterministic
  predictor over a long horizon.
- D-MPC later samples `K` diffusion futures per action candidate and ranks the
  candidate by expected terminal latent L2 distance to the goal:

```text
E_k || z_T^k - z_goal ||_2
```

Do not skip Stage 0. If the diffusion model is not better than LeWM's
deterministic predictor, or at least not clearly capturing useful multimodality,
the planner result will be hard to interpret.

## Cache layout

The repo uses `$STABLEWM_HOME` for checkpoints and datasets. The training scripts
also look at `$LOCAL_DATASET_DIR` when loading datasets. Set both to the same
root while doing this experiment.

```bash
export STABLEWM_HOME=$HOME/.stable_worldmodel
export LOCAL_DATASET_DIR=$STABLEWM_HOME
mkdir -p "$STABLEWM_HOME/datasets" "$STABLEWM_HOME/checkpoints"
```

Expected layout:

```text
$STABLEWM_HOME/
├── datasets/
│   ├── tworoom.h5
│   └── tworoom_latent.lance/
└── checkpoints/
    ├── lewm_tworoom/
    │   ├── config.json
    │   └── weights_epoch_100.pt
    └── latent_diffusion_tworoom/
        ├── config.json
        ├── weights_epoch_100.pt
        └── latent_diffusion_stage0.json
```

Use explicit `.pt` checkpoint paths when a checkpoint folder has multiple epoch
weights:

```bash
export LEWM_CKPT=lewm_tworoom/weights_epoch_100.pt
export DIFF_CKPT=latent_diffusion_tworoom/weights_epoch_100.pt
```

The paths above are relative to `$STABLEWM_HOME/checkpoints`.

## Local setup

From the repo root:

```bash
uv venv --python=3.10
source .venv/bin/activate
uv sync --extra all
```

Check that the package imports:

```bash
uv run python -c "import torch, stable_worldmodel as swm; print(torch.__version__); print(swm.__file__)"
```

On a CUDA machine, also check:

```bash
uv run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

## Get the TwoRoom dataset

The current TwoRoom configs expect:

```text
$STABLEWM_HOME/datasets/tworoom.h5
```

If you already have `tworoom.h5`, copy it there and verify it loads:

```bash
uv run python -c "import stable_worldmodel as swm; ds=swm.data.load_dataset('tworoom.h5', num_steps=4, frameskip=5, keys_to_load=['pixels','action','proprio']); print(len(ds)); print(ds.column_names)"
```

If you collect a fresh dataset with this repo, the existing collector writes a
Lance dataset named `tworoom_expert.lance`:

```bash
uv run python scripts/data/collect_tworooms.py num_traj=1000 world.num_envs=10
```

For collected Lance data, set:

```bash
export TWOROOM_DATA=tworoom_expert.lance
```

For the default HDF5 dataset, set:

```bash
export TWOROOM_DATA=tworoom.h5
```

When `TWOROOM_DATA` is not `tworoom.h5`, add
`data.dataset.name="$TWOROOM_DATA"` to the LeWM, diffusion eval, and planning
commands below, and use `source="$TWOROOM_DATA"` for latent caching.

## Get a pretrained LeWM

You have three workable options.

### Option A: use an existing local checkpoint

Put the checkpoint folder under `$STABLEWM_HOME/checkpoints`:

```text
$STABLEWM_HOME/checkpoints/lewm_tworoom/
├── config.json
└── weights_epoch_100.pt
```

Then:

```bash
export LEWM_CKPT=lewm_tworoom/weights_epoch_100.pt
```

### Option B: use a Hugging Face checkpoint repo

`load_pretrained()` can load a repo id if the repo contains `weights.pt` and
`config.json`.

```bash
export LEWM_CKPT=user-or-org/lewm-tworoom
```

The first load downloads it under:

```text
$STABLEWM_HOME/checkpoints/models--user-or-org--lewm-tworoom/
```

### Option C: train LeWM on TwoRoom

Full GPU training:

```bash
uv run python scripts/train/lewm.py \
  data=tworoom \
  data.dataset.name="$TWOROOM_DATA" \
  output_model_name=lewm_tworoom \
  trainer.max_epochs=100
```

Fast local smoke test:

```bash
uv run python scripts/train/lewm.py \
  data=tworoom \
  data.dataset.name="$TWOROOM_DATA" \
  output_model_name=lewm_tworoom_smoke \
  trainer.max_epochs=1 \
  trainer.accelerator=cpu \
  trainer.devices=1 \
  trainer.precision=32 \
  loader.batch_size=8 \
  num_workers=0 \
  loader.num_workers=0 \
  loader.persistent_workers=false \
  loader.prefetch_factor=null
```

After training, list usable checkpoints:

```bash
find "$STABLEWM_HOME/checkpoints/lewm_tworoom" -name '*.pt' -print
```

Then choose an explicit epoch:

```bash
export LEWM_CKPT=lewm_tworoom/weights_epoch_100.pt
```

## Cache LeWM latents

This encodes the whole dataset once with the frozen LeWM encoder and writes a
new Lance dataset with a `latent` column plus copied action/proprio/state columns
when present.

GPU:

```bash
uv run python scripts/data/cache_latents.py \
  lewm_checkpoint="$LEWM_CKPT" \
  source="$TWOROOM_DATA" \
  out_name=tworoom_latent.lance \
  device=cuda \
  batch_size=256
```

Then set:

```bash
export TWOROOM_LATENT_DATA=tworoom_latent.lance
```

CPU smoke version:

```bash
uv run python scripts/data/cache_latents.py \
  lewm_checkpoint="$LEWM_CKPT" \
  source="$TWOROOM_DATA" \
  out_name=tworoom_latent_smoke.lance \
  device=cpu \
  batch_size=32
```

For the smoke cache, set:

```bash
export TWOROOM_LATENT_DATA=tworoom_latent_smoke.lance
```

Verify:

```bash
uv run python -c "import os, stable_worldmodel as swm; ds=swm.data.load_dataset(os.environ['TWOROOM_LATENT_DATA'], num_steps=11, frameskip=5, keys_to_load=['latent','action']); print(len(ds)); print(ds.column_names)"
```

## Train latent diffusion dynamics

The latent diffusion config defaults to PushT latents, so TwoRoom must override
`data=tworoom_latent`.

Full GPU training:

```bash
uv run python scripts/train/latent_diffusion.py \
  data=tworoom_latent \
  data.dataset.name="$TWOROOM_LATENT_DATA" \
  model.lewm_checkpoint="$LEWM_CKPT" \
  output_model_name=latent_diffusion_tworoom \
  trainer.max_epochs=100
```

Fast local smoke test:

```bash
uv run python scripts/train/latent_diffusion.py \
  data=tworoom_latent \
  data.dataset.name="$TWOROOM_LATENT_DATA" \
  model.lewm_checkpoint="$LEWM_CKPT" \
  output_model_name=latent_diffusion_tworoom_smoke \
  trainer.max_epochs=1 \
  trainer.accelerator=cpu \
  trainer.devices=1 \
  trainer.precision=32 \
  loader.batch_size=8 \
  num_workers=0 \
  loader.num_workers=0 \
  loader.persistent_workers=false \
  loader.prefetch_factor=null
```

Use an explicit epoch for evaluation:

```bash
export DIFF_CKPT=latent_diffusion_tworoom/weights_epoch_100.pt
```

## Stage 0 open-loop evaluation

This is the headline sanity check. It rolls both models open-loop against held
out trajectories and writes a JSON report beside the diffusion checkpoint.

```bash
uv run python scripts/train/eval_latent_diffusion.py \
  data=tworoom \
  data.dataset.name="$TWOROOM_DATA" \
  diffusion_checkpoint="$DIFF_CKPT" \
  num_diffusion_samples=8
```

Default output for an explicit checkpoint path:

```text
$STABLEWM_HOME/checkpoints/latent_diffusion_tworoom/latent_diffusion_stage0.json
```

Key metrics:

- `lewm_step_mse`: LeWM deterministic open-loop latent MSE.
- `diffusion_mean_step_mse`: MSE of the mean over diffusion samples.
- `diffusion_best_step_mse`: oracle best sample MSE among diffusion samples.
- `diffusion_sample_variance`: how much multimodality the sampler is producing.

What to look for:

- `diffusion_mean_step_mse` below `lewm_step_mse` is the cleanest win.
- If the mean is not better, `diffusion_best_step_mse` should still show that
  samples cover plausible futures that LeWM's single predictor smears.
- Final-horizon error matters more than the first few steps.
- If Stage 0 is bad, tune the latent diffusion model before running MPC.

## TwoRoom D-MPC evaluation

Run this only after Stage 0 is credible. The current planning script moves the
model to CUDA, so this is an Oscar/GPU path.

Small planning smoke:

```bash
uv run python scripts/plan/eval_wm.py \
  --config-name=tworoom_diffusion \
  policy="$DIFF_CKPT" \
  eval.dataset_name="$TWOROOM_DATA" \
  eval.num_eval=10 \
  solver.num_samples=64
```

Larger run:

```bash
uv run python scripts/plan/eval_wm.py \
  --config-name=tworoom_diffusion \
  policy="$DIFF_CKPT" \
  eval.dataset_name="$TWOROOM_DATA" \
  eval.num_eval=50 \
  solver.num_samples=300
```

Videos and result text are written beside the diffusion checkpoint folder.

## Oscar setup

Use Oscar for real training. Brown CCV's current Oscar docs are the source of
truth for partitions, accounts, modules, and GPU availability:

```text
https://docs.ccv.brown.edu/oscar/
```

The repo currently has a CPU Submitit launcher for data jobs, but the simplest
path for GPU training is plain `sbatch`.

Recommended Oscar cache root:

```bash
export STABLEWM_HOME=/oscar/scratch/$USER/stable_worldmodel
export LOCAL_DATASET_DIR=$STABLEWM_HOME
mkdir -p "$STABLEWM_HOME/datasets" "$STABLEWM_HOME/checkpoints"
```

Copy data and checkpoints from your laptop if needed:

```bash
rsync -av "$HOME/.stable_worldmodel/datasets/tworoom.h5" \
  "$USER@ssh.ccv.brown.edu:/oscar/scratch/$USER/stable_worldmodel/datasets/"

rsync -av "$HOME/.stable_worldmodel/checkpoints/lewm_tworoom/" \
  "$USER@ssh.ccv.brown.edu:/oscar/scratch/$USER/stable_worldmodel/checkpoints/lewm_tworoom/"
```

Clone and install on Oscar:

```bash
git clone https://github.com/galilai-group/stable-worldmodel.git
cd stable-worldmodel
python -m pip install --user uv  # only needed if `uv` is not already available
uv venv --python=3.10
source .venv/bin/activate
uv sync --extra all
```

Make sure Oscar is using the branch or commit that contains the latent D-MPC
files. If that work is local and not pushed yet, sync the working tree instead
of cloning:

```bash
rsync -av --exclude .venv --exclude .git --exclude __pycache__ \
  ./ "$USER@ssh.ccv.brown.edu:/path/to/stable-worldmodel/"
```

### Oscar batch template

Save this as `scripts/slurm/train_latent_diffusion_tworoom.sbatch` or another
local filename. Adjust partition, account, time, and memory for your allocation.

```bash
#!/bin/bash
#SBATCH -J ldm-tworoom
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
##SBATCH -A <account>

set -euo pipefail

cd /path/to/stable-worldmodel

export MUJOCO_GL=egl
export STABLEWM_HOME=/oscar/scratch/$USER/stable_worldmodel
export LOCAL_DATASET_DIR=$STABLEWM_HOME
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TWOROOM_LATENT_DATA=tworoom_latent.lance

mkdir -p logs "$STABLEWM_HOME/datasets" "$STABLEWM_HOME/checkpoints"

# Use your Oscar Python setup here.
source .venv/bin/activate

export LEWM_CKPT=lewm_tworoom/weights_epoch_100.pt

uv run python scripts/train/latent_diffusion.py \
  data=tworoom_latent \
  data.dataset.name="$TWOROOM_LATENT_DATA" \
  model.lewm_checkpoint="$LEWM_CKPT" \
  output_model_name=latent_diffusion_tworoom \
  trainer.max_epochs=100
```

Submit:

```bash
sbatch scripts/slurm/train_latent_diffusion_tworoom.sbatch
```

### Separate Oscar jobs

Use separate jobs when you want clean logs and easy restarts.

Cache latents:

```bash
#!/bin/bash
#SBATCH -J cache-z-tworoom
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
cd /path/to/stable-worldmodel
source .venv/bin/activate

export MUJOCO_GL=egl
export STABLEWM_HOME=/oscar/scratch/$USER/stable_worldmodel
export LOCAL_DATASET_DIR=$STABLEWM_HOME
export TWOROOM_DATA=tworoom.h5
export LEWM_CKPT=lewm_tworoom/weights_epoch_100.pt

mkdir -p logs "$STABLEWM_HOME/datasets" "$STABLEWM_HOME/checkpoints"

uv run python scripts/data/cache_latents.py \
  lewm_checkpoint="$LEWM_CKPT" \
  source="$TWOROOM_DATA" \
  out_name=tworoom_latent.lance \
  device=cuda \
  batch_size=256
```

Stage 0:

```bash
#!/bin/bash
#SBATCH -J stage0-tworoom
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
cd /path/to/stable-worldmodel
source .venv/bin/activate

export MUJOCO_GL=egl
export STABLEWM_HOME=/oscar/scratch/$USER/stable_worldmodel
export LOCAL_DATASET_DIR=$STABLEWM_HOME
export TWOROOM_DATA=tworoom.h5
export DIFF_CKPT=latent_diffusion_tworoom/weights_epoch_100.pt

mkdir -p logs

uv run python scripts/train/eval_latent_diffusion.py \
  data=tworoom \
  data.dataset.name="$TWOROOM_DATA" \
  diffusion_checkpoint="$DIFF_CKPT" \
  num_diffusion_samples=8
```

Use dependencies if one job must wait for another:

```bash
CACHE_JOB=$(sbatch --parsable scripts/slurm/cache_latents_tworoom.sbatch)
sbatch --dependency=afterok:$CACHE_JOB scripts/slurm/train_latent_diffusion_tworoom.sbatch
```

## Common failure modes

Dataset not found:

- Check that the file is under `$STABLEWM_HOME/datasets`.
- Check that `LOCAL_DATASET_DIR=$STABLEWM_HOME` is exported.
- If using collected Lance data, pass `data.dataset.name=tworoom_expert.lance`
  or set `TWOROOM_DATA=tworoom_expert.lance`.

Checkpoint not found:

- Check that the path is relative to `$STABLEWM_HOME/checkpoints`.
- If a folder has more than one `.pt`, pass the explicit
  `run_name/weights_epoch_N.pt`.
- `config.json` must live beside the `.pt` file.

Action or sequence shape mismatch:

- Keep `frameskip=5`, `history_size=3`, and `horizon=8` aligned across
  `tworoom.yaml`, `tworoom_latent.yaml`, `latent_diffusion.yaml`, and
  `tworoom_diffusion.yaml`.
- TwoRoom D-MPC currently uses `action_block=5`, matching the data frameskip.

CUDA out of memory:

- Lower `loader.batch_size` for training.
- Lower `batch_size` in `cache_latents.py`.
- Lower `solver.num_samples`, `eval.num_eval`, or `num_diffusion_samples` for
  evaluation.

Mac or CPU-only local machine:

- Use CPU smoke commands only.
- Full LeWM, latent caching, diffusion training, and D-MPC should be run on
  Oscar or another CUDA machine.
