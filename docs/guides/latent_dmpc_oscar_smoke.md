---
title: Latent D-MPC Oscar Smoke
summary: Copy-paste commands to continue the Latent D-MPC smoke run on Oscar
---

Use this when the Oscar SSH session closed after the TwoRoom smoke dataset and
LeWM smoke checkpoint were already created.

Expected existing artifacts:

```text
$STABLEWM_HOME/datasets/tworoom_expert.lance/
$STABLEWM_HOME/checkpoints/lewm_tworoom_smoke/weights_epoch_1.pt
$STABLEWM_HOME/checkpoints/lewm_tworoom_smoke/config.json
```

Run these commands from a GPU interactive shell. If needed, get one first:

```bash
interact -q gpu -g 1 -n 8 -m 32g -t 02:00:00
```

## Continue From Step 4

Copy and paste this whole block:

```bash
cd /users/lzheng35/Desktop/brown/world-models/stable-worldmodel
source .venv/bin/activate

export STABLEWM_HOME=/oscar/scratch/$USER/stable_worldmodel
export LOCAL_DATASET_DIR=$STABLEWM_HOME
export TWOROOM_DATA=tworoom_expert.lance
export LEWM_CKPT=lewm_tworoom_smoke/weights_epoch_1.pt
export TWOROOM_LATENT_DATA=tworoom_latent_smoke.lance
export DIFF_CKPT=latent_diffusion_tworoom_smoke/weights_epoch_1.pt

export SLURM_NTASKS=1
export SLURM_NTASKS_PER_NODE=1
export SLURM_CPUS_PER_TASK=8

python -c 'from pathlib import Path; p=Path("scripts/train/lewm.py"); s=p.read_text(); s=s.replace("            config=self.cfg,\n            filename=f'\''weights_epoch_{epoch}.pt'\'',", "            config=self.cfg.model,\n            filename=f'\''weights_epoch_{epoch}.pt'\'',"); p.write_text(s); print("patched lewm.py")'

grep -n "config=self.cfg" scripts/train/lewm.py

python -c 'import json, os; from pathlib import Path; p=Path(os.environ["STABLEWM_HOME"])/"checkpoints"/"lewm_tworoom_smoke"/"config.json"; cfg=json.loads(p.read_text()); p.with_name("config.full.json").write_text(json.dumps(cfg, indent=2)) if "_target_" not in cfg and "model" in cfg else None; p.write_text(json.dumps(cfg["model"], indent=2)) if "_target_" not in cfg and "model" in cfg else None; print(json.loads(p.read_text())["_target_"])'

python scripts/data/cache_latents.py \
  lewm_checkpoint="$LEWM_CKPT" \
  source="$TWOROOM_DATA" \
  out_name="$TWOROOM_LATENT_DATA" \
  device=cuda \
  batch_size=256

python -c "import stable_worldmodel as swm; ds=swm.data.load_dataset('$TWOROOM_LATENT_DATA', num_steps=11, frameskip=5, keys_to_load=['latent','action']); print(len(ds)); print(ds.column_names)"

python scripts/train/latent_diffusion.py \
  data=tworoom_latent \
  data.dataset.name="$TWOROOM_LATENT_DATA" \
  model.lewm_checkpoint="$LEWM_CKPT" \
  output_model_name=latent_diffusion_tworoom_smoke \
  trainer.max_epochs=1 \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  loader.batch_size=32 \
  num_workers=4 \
  loader.num_workers=4

python scripts/train/eval_latent_diffusion.py \
  data=tworoom \
  data.dataset.name="$TWOROOM_DATA" \
  diffusion_checkpoint="$DIFF_CKPT" \
  num_diffusion_samples=4

cat "$STABLEWM_HOME/checkpoints/latent_diffusion_tworoom_smoke/latent_diffusion_stage0.json"
```

The checkpoint repair command should print:

```text
stable_worldmodel.wm.lewm.lewm.LeWM
```

The latent dataset verification should print the dataset length and columns
including:

```text
['latent', 'action']
```

## What This Does

The first Python one-liner patches `scripts/train/lewm.py` so future LeWM
checkpoints save only the model config. The second Python one-liner repairs the
already-created smoke checkpoint by replacing its full Hydra `config.json` with
the nested `model` config that `load_pretrained()` expects.

After that, the block caches frozen LeWM latents, trains the one-epoch latent
diffusion smoke model, and runs Stage 0 open-loop evaluation.

## Next Step

If the smoke run finishes and the Stage 0 JSON exists, scale the run:

```bash
export TWOROOM_DATA=tworoom_expert.lance
export LEWM_CKPT=lewm_tworoom/weights_epoch_100.pt
export TWOROOM_LATENT_DATA=tworoom_latent.lance
export DIFF_CKPT=latent_diffusion_tworoom/weights_epoch_100.pt
```

Then rerun the same pipeline with `output_model_name=lewm_tworoom`,
`out_name=tworoom_latent.lance`, `output_model_name=latent_diffusion_tworoom`,
and `trainer.max_epochs=100`.

For a first diffusion-only sweep, use:

```bash
sbatch scripts/slurm/latent_dmpc_stage0_sweep.sbatch
```

This keeps the LeWM checkpoint and cached latents fixed, trains six diffusion
variants for Stage 0, and writes one `latent_diffusion_stage0.json` per run.
