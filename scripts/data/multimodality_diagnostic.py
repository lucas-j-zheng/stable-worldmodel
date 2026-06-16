"""Measure where a domain's multimodality lives: dynamics vs policy.

Latent D-MPC puts a *diffusion* dynamics model in place of a deterministic
predictor. That only pays off if the thing being modeled is multimodal -- a
deterministic predictor must blur across modes, a sampler can commit to one.
Two distinct distributions could be multimodal, and they motivate diffusion in
*different* parts of the system:

  --mode dynamics : P(z_{t+1} | state_t, action_t).  Multimodality here is what a
                    diffusion *dynamics* model exploits. But given the action,
                    forward dynamics in a deterministic-physics domain are nearly
                    deterministic -- so we expect this to be LOW.
  --mode policy   : P(action_t | state_t).  Multimodality here (many valid actions
                    from one state -- the classic Diffusion Policy / PushT win) is
                    what a diffusion *policy / action-proposal* exploits. A
                    diffusion dynamics model never sees it (it conditions on the
                    action). We expect this to be HIGH on PushT.

Running both on the same latents tells you whether the generative model is on the
right side of the problem. TwoRoom (2026-06-14) and PushT (E6) both lost
closed-loop to the deterministic predictor; if dynamics-multimodality is ~0 while
policy-multimodality is high, that's the mechanism -- and it says move diffusion
to the proposal, not the dynamics.

Method (k-NN in conditioning space, then DETREND):
  1. Build (cond_t, target_t) pairs within episodes per the mode.
  2. Standardize cond; for random anchors gather k nearest neighbors in cond space
     (near-identical conditioning) -- their target spread samples P(target | cond).
  3. Detrend: fit a local-linear map target ~ W.(cond - cond_anchor) and take the
     RESIDUAL. This removes deterministic local sensitivity (which scales with
     neighborhood width, hence dataset sparsity) and isolates true conditional
     stochasticity. A linear fit cannot remove multimodality (it passes between
     branches), so residual bimodality genuinely detects split modes.
  4. Report det_R2 (variance a deterministic local map explains -- high => a
     deterministic model suffices), residual_ratio (leftover stochastic spread),
     and residual_bimodality_fraction (of that leftover, the multimodal share --
     the diffusion edge).

Usage (run via SLURM, not the login node -- this loads a real dataset):
    python scripts/data/multimodality_diagnostic.py \
        --dataset pusht_latent.lance --tag pusht_dyn   --mode dynamics
    python scripts/data/multimodality_diagnostic.py \
        --dataset pusht_latent.lance --tag pusht_policy --mode policy
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.neighbors import BallTree

import stable_worldmodel as swm


def build_pairs(ds, max_frames, mode, rng):
    """Collect within-episode (cond_t, target_t) pairs up to ~max_frames.

    dynamics: cond = [state, action]_t, target = latent_{t+1}.
    policy:   cond = [state]_t,         target = action_t.
    """
    present = set(ds.column_names)
    if mode == 'dynamics':
        cond_cols = [c for c in ('state', 'action') if c in present]
        if 'action' not in cond_cols:
            raise SystemExit(f'dynamics needs action; have {sorted(present)}')
        target_col, shift = 'latent', True
    elif mode == 'policy':
        cond_cols = [c for c in ('state',) if c in present]
        if not cond_cols or 'action' not in present:
            raise SystemExit(f'policy needs state+action; have {sorted(present)}')
        target_col, shift = 'action', False
    else:
        raise SystemExit(f'unknown mode {mode}')
    print(f'[mm] mode={mode} cond={cond_cols} target={target_col} shift={shift}')

    ep_order = rng.permutation(len(ds.lengths))
    conds, targs = [], []
    n = 0
    for ep in ep_order:
        ep = int(ep)
        T = int(ds.lengths[ep])
        if T < 2:
            continue
        data = ds.load_episode(ep)
        parts = [np.asarray(data[c], dtype=np.float32).reshape(T, -1) for c in cond_cols]
        cond = np.concatenate(parts, axis=1)              # (T, C)
        tgt = np.asarray(data[target_col], dtype=np.float32).reshape(T, -1)
        if shift:                                         # target = value at t+1
            conds.append(cond[:T - 1])
            targs.append(tgt[1:T])
            n += T - 1
        else:                                             # target = value at t
            conds.append(cond)
            targs.append(tgt)
            n += T
        if n >= max_frames:
            break
    cond = np.concatenate(conds, axis=0)
    tgt = np.concatenate(targs, axis=0)
    print(f'[mm] {cond.shape[0]} pairs, cond_dim={cond.shape[1]}, '
          f'target_dim={tgt.shape[1]}')
    return cond, tgt


def bimodality_coefficient(x):
    """Sarle's bimodality coefficient. BC > 5/9 (~0.555, uniform) suggests split
    modes; a Gaussian gives ~1/3."""
    x = x - x.mean()
    s = x.std()
    if s < 1e-12 or x.size < 8:
        return 0.0
    z = x / s
    n = x.size
    g1 = np.mean(z ** 3)
    g2 = np.mean(z ** 4) - 3.0
    denom = g2 + 3.0 * ((n - 1) ** 2) / ((n - 2) * (n - 3))
    if abs(denom) < 1e-12:
        return 0.0
    return (g1 ** 2 + 1.0) / denom


def top_pc_projection(mat):
    """Project rows of a centered matrix onto their dominant direction."""
    c = mat - mat.mean(0)
    try:
        _, _, vt = np.linalg.svd(c, full_matrices=False)
        return c @ vt[0]
    except np.linalg.LinAlgError:
        return c[:, 0]


def analyze(cond, target, n_anchors, k, rng):
    """k-NN-in-cond + local-linear detrend stats for P(target | cond)."""
    cmu, csd = cond.mean(0), cond.std(0) + 1e-8
    condn = (cond - cmu) / csd
    marginal_std = float(np.sqrt(target.var(0).mean()))

    tree = BallTree(condn)
    anchors = rng.choice(condn.shape[0], size=min(n_anchors, condn.shape[0]),
                         replace=False)
    k = min(k, condn.shape[0])
    dist, nbr = tree.query(condn[anchors], k=k)

    cond_stds, bc_vals, res_stds, res_bc_vals, det_r2 = [], [], [], [], []
    for a, row in zip(anchors, nbr):
        tloc = target[row]                                # (k, D)
        cond_stds.append(np.sqrt(tloc.var(0).mean()))
        bc_vals.append(bimodality_coefficient(top_pc_projection(tloc)))

        dc = condn[row] - condn[a]
        dc = np.concatenate([dc, np.ones((dc.shape[0], 1))], axis=1)
        W, *_ = np.linalg.lstsq(dc, tloc, rcond=None)
        resid = tloc - dc @ W
        res_stds.append(np.sqrt(resid.var(0).mean()))
        tot = tloc.var(0).mean()
        det_r2.append(1.0 - resid.var(0).mean() / tot if tot > 0 else 0.0)
        res_bc_vals.append(bimodality_coefficient(top_pc_projection(resid)))

    cond_std = float(np.mean(cond_stds))
    res_std = float(np.mean(res_stds))
    bc_vals, res_bc_vals = np.asarray(bc_vals), np.asarray(res_bc_vals)
    return {
        'marginal_std': round(marginal_std, 5),
        'conditional_std': round(cond_std, 5),
        'ratio_cond_over_marginal': round(cond_std / marginal_std if marginal_std else 0, 4),
        'bimodality_fraction': round(float((bc_vals > 5 / 9).mean()), 4),
        'bimodality_coeff_median': round(float(np.median(bc_vals)), 4),
        'det_r2_mean': round(float(np.mean(det_r2)), 4),
        'residual_std': round(res_std, 5),
        'residual_ratio': round(res_std / marginal_std if marginal_std else 0, 4),
        'residual_bimodality_fraction': round(float((res_bc_vals > 5 / 9).mean()), 4),
        'residual_bimodality_coeff_median': round(float(np.median(res_bc_vals)), 4),
        'neighbor_radius_median': round(float(np.median(dist[:, -1])), 4),
        'cond_scale_median': round(float(np.median(np.linalg.norm(condn, axis=1))), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, help='latent .lance name under datasets/')
    ap.add_argument('--tag', required=True, help='label for the output')
    ap.add_argument('--mode', choices=['dynamics', 'policy'], default='dynamics')
    ap.add_argument('--max-frames', type=int, default=200_000)
    ap.add_argument('--n-anchors', type=int, default=4000)
    ap.add_argument('--k', type=int, default=64, help='neighbors per anchor')
    ap.add_argument('--out', default=None, help='output json path')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    ds = swm.data.load_dataset(args.dataset)
    cond, target = build_pairs(ds, args.max_frames, args.mode, rng)

    stats = analyze(cond, target, args.n_anchors, args.k, rng)
    result = {'tag': args.tag, 'dataset': args.dataset, 'mode': args.mode,
              'n_pairs': int(cond.shape[0]), 'cond_dim': int(cond.shape[1]),
              'target_dim': int(target.shape[1]), 'k': min(args.k, cond.shape[0]),
              **stats}
    print('[mm] RESULT', json.dumps(result, indent=2))
    print(f'[mm] mode={args.mode}: det_R2={result["det_r2_mean"]} '
          f'(deterministic-map share; high => deterministic model suffices), '
          f'residual_bimodal_frac={result["residual_bimodality_fraction"]} '
          f'(multimodal share of the leftover -- the diffusion edge for this mode)')

    out = args.out or f'logs/multimodality_{args.tag}.json'
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=2))
    print(f'[mm] wrote {out}')


if __name__ == '__main__':
    main()
