"""Measure how multimodal a domain's next-latent dynamics are.

The latent D-MPC premise is that a *diffusion* dynamics model beats a
deterministic predictor when the true next-state distribution is multimodal
given (state, action) -- because a deterministic predictor is forced to blur
across modes while a sampler can commit to one. TwoRoom said no (deterministic
won), with the hypothesis that TwoRoom dynamics are near-deterministic, the
worst case for a sampler. Before reading anything into the PushT (E6) result we
need to *measure* whether PushT actually exercises the diffusion advantage --
i.e. is P(z_{t+1} | state_t, action_t) meaningfully multimodal?

Method (k-NN in conditioning space, encoder-agnostic question of the dynamics):
  1. Build transitions (cond_t = [state_t, action_t], z_{t+1}) within episodes.
  2. Standardize cond features; build a BallTree on a reference subsample.
  3. For random anchors, gather the k nearest neighbors in cond space -- these
     are transitions from near-identical (state, action), so their spread of
     z_{t+1} is an empirical sample of the conditional next-latent distribution.
  4. Report:
       - ratio = conditional_std / marginal_std. ~0 => deterministic dynamics;
         ->1 => conditioning tells you nothing (max stochastic).
       - bimodality_fraction = fraction of neighborhoods whose top-PC next-latent
         distribution has Sarle's bimodality coefficient > 5/9 (the uniform
         threshold) -- i.e. genuinely split into modes, not just noisy.
       - neighbor cond-radius (sanity: neighbors must actually be close, else
         high spread is just sparse conditioning, not multimodality).

Conditioning is on the *physical* state+action (not the latent history the model
sees) on purpose: this asks whether the *domain's dynamics* are multimodal,
independent of encoder quality -- exactly the "pick domains" question.

Usage (run via SLURM, not the login node -- this loads a real dataset):
    python scripts/data/multimodality_diagnostic.py \
        --dataset pusht_latent.lance --tag pusht
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.neighbors import BallTree

import stable_worldmodel as swm


def build_transitions(ds, max_frames, cond_cols, rng):
    """Collect within-episode (cond_t, z_{t+1}) pairs up to ~max_frames."""
    present = set(ds.column_names)
    use_cols = [c for c in cond_cols if c in present]
    if 'action' not in use_cols:
        raise SystemExit(f'no usable conditioning cols; have {sorted(present)}')
    print(f'[mm] conditioning on {use_cols} (of requested {cond_cols})')

    n_eps = len(ds.lengths)
    ep_order = rng.permutation(n_eps)
    conds, znext = [], []
    n = 0
    for ep in ep_order:
        ep = int(ep)
        if ds.lengths[ep] < 2:
            continue
        data = ds.load_episode(ep)
        z = np.asarray(data['latent'], dtype=np.float32)  # (T, D)
        # cond at step t = concat of the requested per-step features at t
        parts = [np.asarray(data[c], dtype=np.float32).reshape(ds.lengths[ep], -1)
                 for c in use_cols]
        cond = np.concatenate(parts, axis=1)              # (T, C)
        T = z.shape[0]
        conds.append(cond[:T - 1])                        # cond_t,  t = 0..T-2
        znext.append(z[1:T])                              # z_{t+1}
        n += T - 1
        if n >= max_frames:
            break
    conds = np.concatenate(conds, axis=0)
    znext = np.concatenate(znext, axis=0)
    print(f'[mm] {conds.shape[0]} transitions, cond_dim={conds.shape[1]}, '
          f'latent_dim={znext.shape[1]} (from {len(conds)} steps)')
    return conds, znext


def bimodality_coefficient(x):
    """Sarle's bimodality coefficient: (skew^2 + 1) / kurtosis (Fisher, +3/...).

    BC > 5/9 (~0.555, the uniform value) suggests bimodality / split modes.
    """
    x = x - x.mean()
    s = x.std()
    if s < 1e-12 or x.size < 8:
        return 0.0
    z = x / s
    n = x.size
    m3 = np.mean(z ** 3)
    m4 = np.mean(z ** 4)
    # sample-size corrected (DeCarlo); g1 skew, g2 excess kurtosis
    g1 = m3
    g2 = m4 - 3.0
    denom = g2 + 3.0 * ((n - 1) ** 2) / ((n - 2) * (n - 3))
    if abs(denom) < 1e-12:
        return 0.0
    return (g1 ** 2 + 1.0) / denom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, help='latent .lance name under datasets/')
    ap.add_argument('--tag', required=True, help='domain label for the output')
    ap.add_argument('--cond-cols', nargs='+', default=['state', 'action'])
    ap.add_argument('--max-frames', type=int, default=200_000)
    ap.add_argument('--n-anchors', type=int, default=4000)
    ap.add_argument('--k', type=int, default=64, help='neighbors per anchor')
    ap.add_argument('--out', default=None, help='output json path')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    ds = swm.data.load_dataset(args.dataset)
    cond, znext = build_transitions(ds, args.max_frames, args.cond_cols, rng)

    # Standardize conditioning so all features weigh equally in the metric.
    cmu, csd = cond.mean(0), cond.std(0) + 1e-8
    condn = (cond - cmu) / csd

    # Marginal next-latent spread: mean per-dim std (RMS scale of z').
    marginal_std = float(np.sqrt((znext.var(0)).mean()))

    tree = BallTree(condn)
    anchor_idx = rng.choice(condn.shape[0], size=min(args.n_anchors, condn.shape[0]),
                            replace=False)
    k = min(args.k, condn.shape[0])
    dist, nbr = tree.query(condn[anchor_idx], k=k)        # (A, k)

    cond_scale = float(np.median(np.linalg.norm(condn, axis=1)))  # typical |cond|
    nbr_radius = float(np.median(dist[:, -1]))            # median farthest-neighbor

    # Per neighborhood compute TWO families of stats:
    #   raw      = total within-neighborhood spread of z'  (confounded by how
    #              wide the (s,a) neighborhood is -- deterministic sensitivity
    #              leaks in, and that scales with dataset sparsity/dim).
    #   residual = spread AFTER removing a local-linear deterministic map
    #              z' ~ W . (cond - cond_anchor). This subtracts the
    #              deterministic component, isolating genuine conditional
    #              stochasticity -- the only thing a diffusion model can exploit.
    # The residual's bimodality says whether that stochasticity is *multimodal*
    # (diffusion wins) or just unimodal Gaussian noise (diffusion gains nothing).
    cond_stds, bc_vals = [], []
    res_stds, res_bc_vals, det_r2 = [], [], []
    for a, row in zip(anchor_idx, nbr):
        zloc = znext[row]                                 # (k, D)
        cond_stds.append(np.sqrt(zloc.var(0).mean()))
        zc = zloc - zloc.mean(0)
        try:
            _, _, vt = np.linalg.svd(zc, full_matrices=False)
            proj = zc @ vt[0]
        except np.linalg.LinAlgError:
            proj = zc[:, 0]
        bc_vals.append(bimodality_coefficient(proj))

        # Local-linear detrend: regress centered z' on centered cond.
        dc = condn[row] - condn[a]                        # (k, C)
        dc = np.concatenate([dc, np.ones((dc.shape[0], 1))], axis=1)  # + bias
        W, *_ = np.linalg.lstsq(dc, zloc, rcond=None)     # (C+1, D)
        resid = zloc - dc @ W                             # (k, D)
        res_stds.append(np.sqrt(resid.var(0).mean()))
        tot_var = zloc.var(0).mean()
        det_r2.append(1.0 - resid.var(0).mean() / tot_var if tot_var > 0 else 0.0)
        rc = resid - resid.mean(0)
        try:
            _, _, rvt = np.linalg.svd(rc, full_matrices=False)
            rproj = rc @ rvt[0]
        except np.linalg.LinAlgError:
            rproj = rc[:, 0]
        res_bc_vals.append(bimodality_coefficient(rproj))

    cond_std = float(np.mean(cond_stds))
    ratio = cond_std / marginal_std if marginal_std > 0 else 0.0
    bc_vals = np.asarray(bc_vals)
    bimodal_frac = float((bc_vals > 5.0 / 9.0).mean())

    res_std = float(np.mean(res_stds))
    res_ratio = res_std / marginal_std if marginal_std > 0 else 0.0
    res_bc_vals = np.asarray(res_bc_vals)
    res_bimodal_frac = float((res_bc_vals > 5.0 / 9.0).mean())
    det_r2_mean = float(np.mean(det_r2))

    result = {
        'tag': args.tag,
        'dataset': args.dataset,
        'n_transitions': int(cond.shape[0]),
        'cond_dim': int(cond.shape[1]),
        'latent_dim': int(znext.shape[1]),
        'n_anchors': int(len(anchor_idx)),
        'k': int(k),
        'marginal_std': round(marginal_std, 5),
        # raw (confounded by neighborhood width -- kept for continuity)
        'conditional_std': round(cond_std, 5),
        'ratio_cond_over_marginal': round(ratio, 4),
        'bimodality_fraction': round(bimodal_frac, 4),
        'bimodality_coeff_median': round(float(np.median(bc_vals)), 4),
        # detrended (the real signal): residual after a local-linear (s,a) map
        'det_r2_mean': round(det_r2_mean, 4),
        'residual_std': round(res_std, 5),
        'residual_ratio': round(res_ratio, 4),
        'residual_bimodality_fraction': round(res_bimodal_frac, 4),
        'residual_bimodality_coeff_median': round(float(np.median(res_bc_vals)), 4),
        # sanity
        'neighbor_radius_median': round(nbr_radius, 4),
        'cond_scale_median': round(cond_scale, 4),
    }
    print('[mm] RESULT', json.dumps(result, indent=2))
    print(f'[mm] interpretation: det_R2={result["det_r2_mean"]} '
          f'(fraction of next-latent variation a local-linear (s,a) map explains '
          f'-- high => deterministic predictor suffices), residual_ratio='
          f'{result["residual_ratio"]} (leftover stochastic spread), '
          f'residual_bimodal_frac={result["residual_bimodality_fraction"]} '
          f'(of that leftover, fraction that is *multimodal* -- the diffusion edge)')

    out = args.out or f'logs/multimodality_{args.tag}.json'
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=2))
    print(f'[mm] wrote {out}')


if __name__ == '__main__':
    main()
