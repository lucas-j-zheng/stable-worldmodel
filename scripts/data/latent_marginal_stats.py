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
    ap.add_argument(
        '--dump-stats',
        default=None,
        help='npz path for per-dim mean/std (feeds the whitened planning cost: '
        'eval_wm +model_overrides={cost_whiten:true,cost_stats_path:...})',
    )
    args = ap.parse_args()

    cache_dir = os.environ.get('LOCAL_DATASET_DIR', None)
    ds = swm.data.load_dataset(
        args.dataset, cache_dir=cache_dir, keys_to_load=['latent']
    )
    n_ep = min(args.n_episodes, len(ds.lengths))
    lat = np.concatenate(
        [np.asarray(ds.load_episode(i)['latent']) for i in range(n_ep)], axis=0
    )
    per_dim_mean = lat.mean(axis=0)
    per_dim_std = lat.std(axis=0)
    print(
        f'[latstats] {args.dataset}: n={lat.shape[0]} frames, dim={lat.shape[1]} | '
        f'mean_norm={np.linalg.norm(per_dim_mean):.3f} | '
        f'global_std={lat.std():.3f} | '
        f'per-dim std min/med/max='
        f'{per_dim_std.min():.3f}/{np.median(per_dim_std):.3f}/{per_dim_std.max():.3f} | '
        f'|latent| p99={np.percentile(np.abs(lat), 99):.2f} '
        f'(frozen SIGReg ref: mean_norm~0, std~1, p99<~3; clip_sample=6)',
        flush=True,
    )

    # E6.5: `global_std` above is taken over the FLATTENED array, so it mixes
    # within-dim spread with the cross-dim spread of the per-dim means. On a
    # drifted marginal (mean_norm ~10) most of it can be static offset — which
    # would make an "anti-collapse PASS" read on global_std ~1 an artifact.
    # Decompose it: total_var = within_var + mean_spread_var.
    within_var = float((per_dim_std**2).mean())
    mean_spread_var = float(((per_dim_mean - per_dim_mean.mean()) ** 2).mean())
    total_var = within_var + mean_spread_var
    print(
        f'[latstats/decomp] {args.dataset}: '
        f'within_std={np.sqrt(within_var):.3f} (the real anti-collapse number; '
        f'frozen ~0.97) | mean_spread_std={np.sqrt(mean_spread_var):.3f} | '
        f'mean_share_of_global_var={mean_spread_var / max(total_var, 1e-12):.1%} | '
        f'|per-dim mean| med/max='
        f'{np.median(np.abs(per_dim_mean)):.3f}/{np.abs(per_dim_mean).max():.3f} | '
        f'std anisotropy max/min={per_dim_std.max() / max(per_dim_std.min(), 1e-6):.1f}x',
        flush=True,
    )

    if args.dump_stats:
        np.savez(args.dump_stats, mean=per_dim_mean, std=per_dim_std)
        print(f'[latstats] wrote per-dim stats -> {args.dump_stats}', flush=True)


if __name__ == '__main__':
    main()
