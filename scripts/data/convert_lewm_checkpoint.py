"""Remap a LeWM checkpoint saved with old HF ViTModel naming to current code.

The pretrained quentinll/lewm-* encoders were saved with an older `transformers`
where ViT attention blocks used `encoder.layer.N.attention.attention.{query,key,
value}` / `intermediate.dense` / `output.dense`. Current code builds the flattened
naming `encoder.layers.N.attention.{q,k,v,o}_proj` / `mlp.fc1` / `mlp.fc2`. Only
the 12 transformer blocks differ; everything else matches.

Usage (run via SLURM, not login node):
    python scripts/data/convert_lewm_checkpoint.py \
        --src-repo quentinll/lewm-pusht --out lewm_pusht
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
import stable_worldmodel as swm
from hydra.utils import instantiate
from omegaconf import OmegaConf


def remap_key(k: str) -> str:
    k = k.replace('encoder.encoder.layer.', 'encoder.layers.')
    # Attention block: order matters (attention.output before bare output).
    k = k.replace('.attention.attention.query', '.attention.q_proj')
    k = k.replace('.attention.attention.key', '.attention.k_proj')
    k = k.replace('.attention.attention.value', '.attention.v_proj')
    k = k.replace('.attention.output.dense', '.attention.o_proj')
    k = k.replace('.intermediate.dense', '.mlp.fc1')
    k = k.replace('.output.dense', '.mlp.fc2')
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src-repo', required=True, help='HF repo id or local ckpt dir')
    ap.add_argument('--out', required=True, help='output run name under checkpoints/')
    args = ap.parse_args()

    ckpt_root = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'))

    # Resolve the downloaded HF model dir (config.json + weights.pt).
    from huggingface_hub import snapshot_download
    src_dir = Path(snapshot_download(args.src_repo))
    config = json.loads((src_dir / 'config.json').read_text())
    state = torch.load(src_dir / 'weights.pt', map_location='cpu')
    print(f'[convert] {len(state)} source params from {args.src_repo}')

    # Build the model the current code expects, to get target key names.
    model = instantiate(OmegaConf.create(config))
    target_keys = set(model.state_dict().keys())

    new_state = {remap_key(k): v for k, v in state.items()}

    src_only = set(new_state) - target_keys
    tgt_only = target_keys - set(new_state)
    print(f'[convert] after remap: {len(src_only)} unexpected, {len(tgt_only)} missing')
    if src_only:
        print('  unexpected (sample):', sorted(src_only)[:6])
    if tgt_only:
        print('  missing (sample):', sorted(tgt_only)[:6])

    # Strict load — must be exact.
    model.load_state_dict(new_state, strict=True)
    print('[convert] strict load OK')

    # Strict load already validates architecture + shapes match exactly; this
    # forward pass just confirms the remapped encoder runs end to end.
    model.eval()
    with torch.no_grad():
        x = torch.randn(8, 3, config['encoder']['image_size'],
                        config['encoder']['image_size'])
        emb = model.encode({'pixels': x})['emb']
        print(f'[convert] forward OK: emb shape={tuple(emb.shape)} '
              f'finite={torch.isfinite(emb).all().item()}')

    out_dir = ckpt_root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / 'weights.pt')
    shutil.copy(src_dir / 'config.json', out_dir / 'config.json')
    print(f'[convert] saved -> {out_dir}/weights.pt (+ config.json)')


if __name__ == '__main__':
    main()
