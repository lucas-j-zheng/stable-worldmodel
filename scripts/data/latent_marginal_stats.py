"""Latent-marginal drift check (ARC 5): did e2e fine-tuning break the N(0,I)
calibration the D-MPC stack assumes?

The frozen LeWM/SIGReg latents are ~N(0, I) by construction; the diffusion
prior (cosine terminal), static clip_sample=6.0, and CEM cost geometry all
lean on that. Joint fine-tuning trains WITHOUT the SIGReg regularizer, so the
marginal can drift. If the e2e latents' scale/mean has moved far from (0, 1),
that explains why every e2e checkpoint floors in closed loop (both objectives,
both envs) despite better bimodality: the planner's assumptions broke, not the
dynamics knowledge.

Usage: python scripts/data/latent_marginal_stats.py --dataset <name.lance> [--n-episodes 50]
(plain argparse — no hydra; the data-config's MULTIRUN mode swallowed stdout)
"""

import argparse
import os

os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np

import stable_worldmodel as swm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--n-episodes', type=int, default=50)
    args = ap.parse_args()

    cache_dir = os.environ.get('LOCAL_DATASET_DIR', None)
    ds = swm.data.load_dataset(
        args.dataset, cache_dir=cache_dir, keys_to_load=['latent']
    )
    n_ep = min(args.n_episodes, len(ds.lengths))
    lat = np.concatenate(
        [np.asarray(ds.load_episode(i)['latent']) for i in range(n_ep)], axis=0
    )
    per_dim_std = lat.std(axis=0)
    print(
        f'[latstats] {args.dataset}: n={lat.shape[0]} frames, dim={lat.shape[1]} | '
        f'mean_norm={np.linalg.norm(lat.mean(axis=0)):.3f} | '
        f'global_std={lat.std():.3f} | '
        f'per-dim std min/med/max='
        f'{per_dim_std.min():.3f}/{np.median(per_dim_std):.3f}/{per_dim_std.max():.3f} | '
        f'|latent| p99={np.percentile(np.abs(lat), 99):.2f} '
        f'(frozen SIGReg ref: mean_norm~0, std~1, p99<~3; clip_sample=6)',
        flush=True,
    )


if __name__ == '__main__':
    main()
