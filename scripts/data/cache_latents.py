"""Cache frozen-encoder latents for an offline trajectory dataset.

D-MPC component 0: encode the whole dataset once with the frozen LeWM/SIGReg
encoder and write a derived dataset whose ``pixels`` are replaced by a 192-d
``latent`` column. Everything downstream (diffusion dynamics training) then
operates on ``z`` and never re-runs the vision encoder.

The latent marginal is ~N(0, I) by construction (SIGReg), so latents are stored
as-is -- no normalization.

Example
-------
    python scripts/data/cache_latents.py \
        lewm_checkpoint=lewm/weights_epoch_100.pt \
        source=pusht_expert_train.lance \
        out_name=pusht_expert_train_latent.lance
"""

import os

os.environ.setdefault('MUJOCO_GL', 'egl')

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torchvision.transforms import v2 as transforms

import stable_pretraining as spt
import stable_worldmodel as swm
from stable_worldmodel.data import LanceWriter, get_cache_dir


def img_transform(img_size: int, dtype=torch.float32):
    """ImageNet-normalized resize, matching scripts/plan/eval_wm.py."""
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(dtype, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=img_size),
        ]
    )


@torch.no_grad()
def encode_episode(lewm, pixels: np.ndarray, transform, device, batch_size):
    """Encode one episode's raw pixels (T, H, W, C) -> latents (T, D)."""
    frames = torch.stack([transform(frame) for frame in pixels])  # (T, C, H, W)
    enc_dtype = next(lewm.encoder.parameters()).dtype
    latents = []
    for start in range(0, frames.shape[0], batch_size):
        chunk = frames[start : start + batch_size].to(device=device, dtype=enc_dtype)
        # lewm.encode expects (B, T, C, H, W); use B=1 and the chunk as time.
        emb = lewm.encode({'pixels': chunk.unsqueeze(0)})['emb'][0]
        latents.append(emb.float().cpu())
    return torch.cat(latents, dim=0).numpy()


@hydra.main(version_base=None, config_path='./config', config_name='cache_latents')
def run(cfg: DictConfig):
    device = torch.device(cfg.device)

    lewm = swm.wm.utils.load_pretrained(cfg.lewm_checkpoint)
    lewm = lewm.to(device).eval()
    lewm.requires_grad_(False)
    if hasattr(lewm, 'interpolate_pos_encoding'):
        lewm.interpolate_pos_encoding = True

    cache_dir = os.environ.get('LOCAL_DATASET_DIR', None)
    keys_to_copy = list(cfg.keys_to_copy)

    # ``keys_to_copy`` are optional. ``load_dataset`` validates ``keys_to_load``
    # eagerly and raises if any are absent, so requesting them up front would
    # crash before the skip logic. Probe the available columns first (no key
    # filter loads column names, not data), then request only the survivors.
    probe = swm.data.load_dataset(cfg.source, cache_dir=cache_dir)
    present = set(probe.column_names)
    copy_cols = [c for c in keys_to_copy if c in present]
    missing = [c for c in keys_to_copy if c not in present]
    if missing:
        print(f'[cache_latents] skipping absent columns: {missing}')

    dataset = swm.data.load_dataset(
        cfg.source,
        cache_dir=cache_dir,
        keys_to_load=['pixels', *copy_cols],
    )

    transform = img_transform(cfg.img_size)
    out_path = get_cache_dir(cache_dir, sub_folder='datasets') / cfg.out_name
    print(
        f'[cache_latents] {cfg.source} -> {out_path} '
        f'({len(dataset.lengths)} episodes, columns: latent + {copy_cols})'
    )

    n_episodes = len(dataset.lengths)

    def episode_iter():
        for ep in range(n_episodes):
            ep_data = dataset.load_episode(ep)
            latents = encode_episode(
                lewm, ep_data['pixels'], transform, device, cfg.batch_size
            )
            out = {'latent': [row for row in latents]}
            for col in copy_cols:
                out[col] = [row for row in np.asarray(ep_data[col])]
            if (ep + 1) % cfg.get('log_every', 50) == 0:
                print(f'[cache_latents] encoded {ep + 1}/{n_episodes} episodes')
            yield out

    with LanceWriter(str(out_path), mode='overwrite') as writer:
        writer.write_episodes(episode_iter())

    print(f'[cache_latents] done. wrote {out_path}')


if __name__ == '__main__':
    run()
