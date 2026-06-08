"""Diagnose whether LEWM latent L2 tracks temporal distance.

Example:
    python scripts/diagnostics/lewm_latent_distance.py \
        --checkpoint lewm/weights_epoch_100.pt \
        --dataset pusht_expert_train.lance \
        --frameskip 5
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Measure whether LEWM latent L2 behaves like temporal value.'
        )
    )
    parser.add_argument(
        '--checkpoint',
        required=True,
        help=(
            'Checkpoint name/path accepted by swm.wm.utils.load_pretrained. '
            'For run folders with multiple .pt files, pass the file '
            'explicitly.'
        ),
    )
    parser.add_argument(
        '--dataset',
        default='pusht_expert_train.lance',
        help='Dataset name/path accepted by swm.data.load_dataset.',
    )
    parser.add_argument('--dataset-format', default=None)
    parser.add_argument(
        '--cache-dir',
        default=None,
        help=(
            'Optional STABLEWM_HOME-style cache root for datasets/checkpoints.'
        ),
    )
    parser.add_argument('--pixels-key', default='pixels')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument(
        '--frameskip',
        type=int,
        default=1,
        help='Dataset frameskip used while loading episodes.',
    )
    parser.add_argument('--num-episodes', type=int, default=32)
    parser.add_argument('--max-frames-per-episode', type=int, default=256)
    parser.add_argument('--pairs-per-episode', type=int, default=512)
    parser.add_argument('--goals-per-episode', type=int, default=32)
    parser.add_argument(
        '--min-goal-distance',
        type=int,
        default=10,
        help='Minimum sampled start-goal separation in raw environment steps.',
    )
    parser.add_argument('--num-bins', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--device',
        default='auto',
        help='auto, cpu, cuda, cuda:0, mps, etc.',
    )
    parser.add_argument(
        '--bf16',
        action='store_true',
        help='Run model weights/inputs in bfloat16 where supported.',
    )
    parser.add_argument(
        '--monotonic-tolerance',
        type=float,
        default=0.0,
        help=(
            'Allowed adjacent increase when checking goal-distance '
            'monotonicity.'
        ),
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Optional JSON output path.',
    )
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device != 'auto':
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device('cuda')
    mps_backend = getattr(torch.backends, 'mps', None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def preprocess_pixels(pixels, img_size: int) -> torch.Tensor:
    pixels = torch.as_tensor(pixels)
    if pixels.ndim == 3:
        pixels = pixels.unsqueeze(0)
    if pixels.ndim != 4:
        raise ValueError(
            'Expected pixels with shape (T, C, H, W) or (T, H, W, C), '
            f'got {tuple(pixels.shape)}'
        )

    if pixels.shape[-1] in (1, 3, 4) and pixels.shape[1] not in (1, 3, 4):
        pixels = pixels.permute(0, 3, 1, 2)
    if pixels.shape[1] == 1:
        pixels = pixels.repeat(1, 3, 1, 1)
    if pixels.shape[1] == 4:
        pixels = pixels[:, :3]
    if pixels.shape[1] != 3:
        raise ValueError(
            f'Expected 1/3/4 image channels, got {pixels.shape[1]}'
        )

    pixels = pixels.float()
    if pixels.numel() and pixels.max() > 2.0:
        pixels = pixels / 255.0

    pixels = F.interpolate(
        pixels,
        size=(img_size, img_size),
        mode='bilinear',
        align_corners=False,
        antialias=True,
    )
    mean = torch.tensor(IMAGENET_MEAN, dtype=pixels.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=pixels.dtype).view(1, 3, 1, 1)
    return (pixels - mean) / std


class PixelTransform:
    def __init__(self, pixels_key: str, img_size: int):
        self.pixels_key = pixels_key
        self.img_size = img_size

    def __call__(self, sample: dict) -> dict:
        sample = dict(sample)
        sample[self.pixels_key] = preprocess_pixels(
            sample[self.pixels_key], self.img_size
        )
        return sample


def select_frames(
    pixels: torch.Tensor,
    max_frames: int,
) -> tuple[torch.Tensor, np.ndarray]:
    total = pixels.shape[0]
    if max_frames <= 0 or total <= max_frames:
        frame_indices = np.arange(total, dtype=np.int64)
    else:
        frame_indices = np.linspace(
            0, total - 1, num=max_frames, dtype=np.int64
        )
        frame_indices = np.unique(frame_indices)
    return pixels[torch.as_tensor(frame_indices)], frame_indices


def encode_pixels(
    model: torch.nn.Module,
    pixels: torch.Tensor,
    *,
    pixels_key: str,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    latents = []
    with torch.inference_mode():
        for start in range(0, pixels.shape[0], batch_size):
            batch = pixels[start : start + batch_size].to(
                device, non_blocking=device.type == 'cuda'
            )
            output = model.encode({'pixels': batch.unsqueeze(1)})
            emb = output['emb'][:, 0]
            if emb.ndim > 2:
                emb = emb.reshape(emb.shape[0], -1)
            latents.append(emb.detach().float().cpu())
    return torch.cat(latents, dim=0).numpy()


def load_model(args: argparse.Namespace, device: torch.device):
    import stable_worldmodel as swm

    model = swm.wm.utils.load_pretrained(
        args.checkpoint,
        cache_dir=args.cache_dir,
    )
    model = model.to(device)
    if args.bf16:
        model = model.to(torch.bfloat16)
    model.eval()
    model.requires_grad_(False)
    return model


def load_dataset(args: argparse.Namespace):
    import stable_worldmodel as swm

    dataset = swm.data.load_dataset(
        args.dataset,
        cache_dir=args.cache_dir,
        format=args.dataset_format,
        frameskip=args.frameskip,
        num_steps=1,
        keys_to_load=[args.pixels_key],
        transform=PixelTransform(args.pixels_key, args.img_size),
    )
    return dataset


def choose_episodes(
    lengths: np.ndarray,
    *,
    num_episodes: int,
    frameskip: int,
    rng: np.random.Generator,
) -> np.ndarray:
    min_raw_length = max(frameskip + 1, 2)
    candidates = np.flatnonzero(lengths >= min_raw_length)
    if candidates.size == 0:
        raise ValueError('No episodes are long enough for temporal pairs.')
    count = min(num_episodes, candidates.size)
    return np.sort(rng.choice(candidates, size=count, replace=False))


def format_metric(value) -> str:
    if value is None:
        return 'n/a'
    return f'{float(value):.4f}'


def print_summary(results: dict) -> None:
    pairwise = results['pairwise_temporal']
    goal = results['goal_conditioned']

    print('\nPairwise latent L2 vs temporal distance')
    print(f"  pairs: {pairwise['n']}")
    print(f"  pearson:  {format_metric(pairwise['pearson'])}")
    print(f"  spearman: {format_metric(pairwise['spearman'])}")
    print(f"  R^2 temporal_from_l2: {format_metric(pairwise['r2'])}")

    print('\nFixed-goal distance-to-goal monotonicity')
    print(f"  segments: {goal['segments']}")
    print(f"  samples: {goal['n']}")
    print(f"  pearson:  {format_metric(goal['pearson'])}")
    print(f"  spearman: {format_metric(goal['spearman'])}")
    print(
        '  adjacent decrease fraction: '
        f"{format_metric(goal['monotone_decrease_fraction'])}"
    )
    print(
        '  per-segment spearman mean: '
        f"{format_metric(goal['segment_spearman_mean'])}"
    )


def main() -> None:
    args = parse_args()
    from stable_worldmodel.wm import diagnostics

    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)

    print(f'Loading LEWM checkpoint: {args.checkpoint}')
    model = load_model(args, device)

    print(f'Loading dataset: {args.dataset}')
    dataset = load_dataset(args)
    episode_indices = choose_episodes(
        dataset.lengths,
        num_episodes=args.num_episodes,
        frameskip=dataset.frameskip,
        rng=rng,
    )

    all_latents = []
    all_times = []
    all_pairs = []
    all_segments = []
    offset = 0
    skipped = 0
    min_goal_frames = max(
        1, math.ceil(args.min_goal_distance / dataset.frameskip)
    )

    for ep_idx in tqdm(episode_indices, desc='Encoding episodes'):
        episode = dataset.load_episode(int(ep_idx))
        pixels = episode[args.pixels_key]
        pixels, frame_indices = select_frames(
            pixels, args.max_frames_per_episode
        )
        if pixels.shape[0] < 2:
            skipped += 1
            continue

        latents = encode_pixels(
            model,
            pixels,
            pixels_key=args.pixels_key,
            device=device,
            batch_size=args.batch_size,
        )
        times = frame_indices.astype(np.float64) * float(dataset.frameskip)

        pairs = diagnostics.sample_temporal_pairs(
            latents.shape[0], args.pairs_per_episode, rng
        )
        segments = diagnostics.sample_goal_segments(
            latents.shape[0],
            args.goals_per_episode,
            rng,
            min_distance=min_goal_frames,
        )

        all_latents.append(latents)
        all_times.append(times)
        all_pairs.append(pairs + offset)
        all_segments.append(segments + offset)
        offset += latents.shape[0]

    if not all_latents:
        raise ValueError('No episodes produced at least two encoded frames.')

    latents = np.concatenate(all_latents, axis=0)
    times = np.concatenate(all_times, axis=0)
    pairs = (
        np.concatenate(all_pairs, axis=0)
        if all_pairs
        else np.empty((0, 2), dtype=np.int64)
    )
    segments = (
        np.concatenate(all_segments, axis=0)
        if all_segments
        else np.empty((0, 2), dtype=np.int64)
    )

    results = {
        'config': {
            **vars(args),
            'device': str(device),
            'effective_frameskip': int(dataset.frameskip),
            'min_goal_frames': int(min_goal_frames),
        },
        'num_encoded_episodes': len(all_latents),
        'num_skipped_episodes': int(skipped),
        'num_encoded_frames': int(latents.shape[0]),
        'latent_dim': int(latents.shape[1]),
        'pairwise_temporal': diagnostics.temporal_distance_metrics(
            latents,
            times,
            pairs,
            num_bins=args.num_bins,
        ),
        'goal_conditioned': diagnostics.goal_monotonicity_metrics(
            latents,
            times,
            segments,
            tolerance=args.monotonic_tolerance,
            num_bins=args.num_bins,
        ),
    }

    print_summary(results)

    if args.output is not None:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('w') as f:
            json.dump(results, f, indent=2)
        print(f'\nWrote metrics to {output}')


if __name__ == '__main__':
    main()
