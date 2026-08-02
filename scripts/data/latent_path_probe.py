"""Which MEASUREMENT PATH does the latent mean drift live in? (E6.5 / P-path)

The E6.0a "mean drift" finding — fine-tuned latents at mean_norm ~10 vs frozen
~0 — is the mechanism the whole E6.5 cell was built on. The moment-penalty arms
then produced a contradiction: the penalty drove the TRAINING-batch mean to
~0.14 (fit/mean_loss 1e-4) while the offline cache of the same checkpoint still
measured 9.87. A per-sample encoder's batch mean is an unbiased estimate of its
population mean, so those two numbers cannot both describe the same
(encoder, input distribution, module mode). Something differs across the paths:

  train  scripts/train/latent_diffusion.py -> spt dt.transforms
         (ToImage(ImageNet) -> Resize), module in TRAIN mode
  cache  scripts/data/cache_latents.py -> torchvision v2
         (ToImage -> ToDtype -> Normalize(ImageNet) -> Resize), EVAL mode
  eval   scripts/plan/eval_wm.py -> its own img_transform, on 224x224 frames
         rendered live by the env (never resized up from stored pixels)

This encodes the SAME frames through each path and reports the per-dim mean and
std of the result, for a fine-tuned encoder and (as control) the frozen one.

  - train-mode vs eval-mode on the same transform isolates MODULE MODE.
  - train-transform vs cache-transform in the same mode isolates PREPROCESSING.
  - a frozen encoder that is path-INsensitive while the fine-tuned one is
    path-SENSITIVE means fine-tuning destroyed preprocessing robustness — and
    the "drift" is then a train/inference mismatch, not a property of e2e.

Run via SLURM (loads a real dataset + encoder).
"""

import argparse
import os

os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import torch
import stable_pretraining as spt
from stable_pretraining import data as dt
from torchvision.transforms import v2 as tv_transforms


def train_path_transform(img_size=224):
    """Exactly what scripts/train/latent_diffusion.py builds (dict-in/dict-out)."""
    imagenet_stats = dt.dataset_stats.ImageNet
    return dt.transforms.Compose(
        dt.transforms.ToImage(
            **imagenet_stats, source='pixels', target='pixels'
        ),
        dt.transforms.Resize(img_size, source='pixels', target='pixels'),
    )


def cache_path_transform(img_size=224):
    """Exactly what scripts/data/cache_latents.py builds (per-frame tensor)."""
    return tv_transforms.Compose(
        [
            tv_transforms.ToImage(),
            tv_transforms.ToDtype(torch.float32, scale=True),
            tv_transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            tv_transforms.Resize(size=img_size),
        ]
    )


def load_encoder(name, device):
    from stable_worldmodel.wm.utils import load_pretrained

    model = load_pretrained(name)
    enc = getattr(model, 'lewm', model)
    return enc.to(device)


@torch.no_grad()
def encode_train_path(enc, frames_hwc, transform, device, chunk=32):
    """frames: (T,H,W,C) uint8 -> (T,D). Mirrors the training dataset pipeline."""
    out = []
    for i in range(0, len(frames_hwc), chunk):
        block = frames_hwc[i : i + chunk]
        # The training loader hands the encoder a (B, T, C, H, W) stack; here one
        # "episode chunk" plays the role of the time axis.
        batch = {'pixels': torch.from_numpy(np.asarray(block))}
        px = transform(batch)['pixels'].to(device)
        if px.ndim == 4:
            px = px.unsqueeze(0)
        out.append(enc.encode({'pixels': px})['emb'].squeeze(0).float().cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def encode_cache_path(enc, frames_hwc, transform, device, chunk=32):
    out = []
    for i in range(0, len(frames_hwc), chunk):
        block = frames_hwc[i : i + chunk]
        px = torch.stack([transform(f) for f in block]).to(device)
        out.append(
            enc.encode({'pixels': px.unsqueeze(0)})['emb'].squeeze(0).float().cpu()
        )
    return torch.cat(out).numpy()


def report(tag, lat):
    per_dim_mean = lat.mean(axis=0)
    per_dim_std = lat.std(axis=0)
    within = float(np.sqrt((per_dim_std**2).mean()))
    print(
        f'[path] {tag}: n={lat.shape[0]} dim={lat.shape[1]} | '
        f'mean_norm={np.linalg.norm(per_dim_mean):.3f} | '
        f'within_std={within:.3f} | '
        f'|per-dim mean| med/max='
        f'{np.median(np.abs(per_dim_mean)):.3f}/{np.abs(per_dim_mean).max():.3f}',
        flush=True,
    )
    return per_dim_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoder', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--dataset', default='pusht_expert_train.h5')
    ap.add_argument('--n-episodes', type=int, default=10)
    ap.add_argument('--img-size', type=int, default=224)
    args = ap.parse_args()

    import stable_worldmodel as swm

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    enc = load_encoder(args.encoder, device)
    ds = swm.data.load_dataset(
        args.dataset, cache_dir=os.environ.get('LOCAL_DATASET_DIR', None)
    )

    frames = np.concatenate(
        [
            np.asarray(ds.load_episode(i)['pixels'])
            for i in range(min(args.n_episodes, len(ds.lengths)))
        ],
        axis=0,
    )
    if frames.shape[1] in (1, 3):  # TCHW -> THWC for the per-frame transforms
        frames = frames.transpose(0, 2, 3, 1)
    print(f'[path] {args.tag}: {frames.shape[0]} frames, raw {frames.shape[1:]}')

    tr_tf = train_path_transform(args.img_size)
    ca_tf = cache_path_transform(args.img_size)

    enc.train()
    m_train_mode = report(
        f'{args.tag}/train-transform/TRAIN-mode',
        encode_train_path(enc, frames, tr_tf, device),
    )
    enc.eval()
    m_train_tf = report(
        f'{args.tag}/train-transform/eval-mode',
        encode_train_path(enc, frames, tr_tf, device),
    )
    m_cache_tf = report(
        f'{args.tag}/cache-transform/eval-mode',
        encode_cache_path(enc, frames, ca_tf, device),
    )

    print(
        f'[path] {args.tag} ATTRIBUTION: '
        f'mode effect (train vs eval, same transform) = '
        f'{np.linalg.norm(m_train_mode - m_train_tf):.3f} | '
        f'transform effect (train-tf vs cache-tf, same mode) = '
        f'{np.linalg.norm(m_train_tf - m_cache_tf):.3f}',
        flush=True,
    )


if __name__ == '__main__':
    main()
