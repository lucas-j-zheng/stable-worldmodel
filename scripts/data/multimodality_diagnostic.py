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

# stable_worldmodel is imported lazily inside the dataset path so --self-test
# (pure synthetic numpy) runs anywhere, including the macOS venv where the swm
# import chain can break on optional deps.


def build_pairs(ds, max_frames, mode, rng, chunk=8):
    """Collect within-episode (cond_t, target_t) pairs up to ~max_frames.

    dynamics     : cond = [state, action]_t,  target = latent_{t+1}.
    policy       : cond = [state]_t,          target = action_t.
    policy_chunk : cond = latent_t (the partial obs the policy sees),
                   target = flattened action chunk [a_t .. a_{t+H-1}].
                   This is the faithful Diffusion-Policy setting: multimodality
                   lives in obs-conditioned action *sequences*, not single-step
                   full-state actions. Conditioning on z (image-derived, partial)
                   instead of the full physical state keeps the ambiguity DP
                   exploits; chunking captures sequence-level branching.
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
    elif mode == 'policy_chunk':
        if 'latent' not in present or 'action' not in present:
            raise SystemExit(f'policy_chunk needs latent+action; have {sorted(present)}')
    else:
        raise SystemExit(f'unknown mode {mode}')

    if mode == 'policy_chunk':
        print(f'[mm] mode=policy_chunk cond=latent target=action_chunk(H={chunk})')
        ep_order = rng.permutation(len(ds.lengths))
        conds, targs = [], []
        n = 0
        for ep in ep_order:
            ep = int(ep)
            T = int(ds.lengths[ep])
            if T <= chunk:
                continue
            data = ds.load_episode(ep)
            z = np.asarray(data['latent'], dtype=np.float32).reshape(T, -1)
            a = np.asarray(data['action'], dtype=np.float32).reshape(T, -1)
            for t in range(0, T - chunk):
                conds.append(z[t])
                targs.append(a[t:t + chunk].reshape(-1))
            n += T - chunk
            if n >= max_frames:
                break
        cond = np.asarray(conds, dtype=np.float32)
        tgt = np.asarray(targs, dtype=np.float32)
        print(f'[mm] {cond.shape[0]} pairs, cond_dim={cond.shape[1]} (latent), '
              f'target_dim={tgt.shape[1]} ({chunk}x action)')
        return cond, tgt

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


def pca_reduce(x, n_comp):
    """Standardize then project onto top-n_comp PCs. High-dim k-NN (192-d latent)
    suffers distance concentration; reduce so neighborhoods are meaningful."""
    xs = (x - x.mean(0)) / (x.std(0) + 1e-8)
    _, s, vt = np.linalg.svd(xs - xs.mean(0), full_matrices=False)
    k = min(n_comp, vt.shape[0])
    var = (s ** 2)
    kept = float(var[:k].sum() / var.sum())
    print(f'[mm] PCA cond {x.shape[1]}->{k} dims, variance kept={kept:.3f}')
    return xs @ vt[:k].T


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


def synthetic_control(rng, n=8000, cond_dim=4, tgt_dim=16, noise=0.15, sep=3.0):
    """Positive/negative controls with KNOWN structure, to validate the metric.

    All three share a deterministic linear trend target = cond @ W (+ structure):
      unimodal : + Gaussian noise          -> expect residual_bimodal ~ 0
      bimodal  : + Bernoulli(+/-sep) branch -> expect residual_bimodal HIGH
      trimodal : + 3-way {-sep,0,+sep} mix  -> expect residual_bimodal HIGH-ish
    If the detrended metric flags the bi/tri cases but not the unimodal one, the
    ~0 readings on TwoRoom/PushT are trustworthy (data really is unimodal), not an
    artifact of an over-aggressive detrend.
    """
    cond = rng.standard_normal((n, cond_dim)).astype(np.float32)
    W = rng.standard_normal((cond_dim, tgt_dim)).astype(np.float32)
    trend = cond @ W
    dirn = rng.standard_normal(tgt_dim).astype(np.float32)
    dirn /= np.linalg.norm(dirn)
    out = {}
    base = trend + noise * rng.standard_normal((n, tgt_dim)).astype(np.float32)
    out['unimodal'] = base
    b = rng.integers(0, 2, n) * 2 - 1                      # +/-1
    out['bimodal'] = base + sep * b[:, None] * dirn[None, :]
    t = rng.integers(0, 3, n) - 1                          # {-1,0,1}
    out['trimodal'] = base + sep * t[:, None] * dirn[None, :]
    return cond, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default=None, help='latent .lance name under datasets/')
    ap.add_argument('--self-test', action='store_true',
                    help='run synthetic positive/negative controls (no dataset)')
    ap.add_argument('--tag', default='mm', help='label for the output')
    ap.add_argument('--mode', choices=['dynamics', 'policy', 'policy_chunk'],
                    default='dynamics')
    ap.add_argument('--chunk', type=int, default=8, help='action-chunk H (policy_chunk)')
    ap.add_argument('--cond-pca', type=int, default=0,
                    help='if >0, PCA-reduce cond to this many dims before k-NN')
    ap.add_argument('--max-frames', type=int, default=200_000)
    ap.add_argument('--n-anchors', type=int, default=4000)
    ap.add_argument('--k', type=int, default=64, help='neighbors per anchor')
    ap.add_argument('--out', default=None, help='output json path')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    if args.self_test:
        cond, variants = synthetic_control(rng)
        print('[mm] SELF-TEST (expect: unimodal residual_bimodal~0, bi/tri HIGH)')
        for name, tgt in variants.items():
            s = analyze(cond, tgt, args.n_anchors, args.k, rng)
            print(f'  {name:9s}: det_R2={s["det_r2_mean"]:.3f}  '
                  f'residual_bimodal_frac={s["residual_bimodality_fraction"]:.3f}  '
                  f'residual_bc_median={s["residual_bimodality_coeff_median"]:.3f}')
        return

    if not args.dataset:
        raise SystemExit('provide --dataset or --self-test')
    import stable_worldmodel as swm
    ds = swm.data.load_dataset(args.dataset)
    cond, target = build_pairs(ds, args.max_frames, args.mode, rng, chunk=args.chunk)

    cond_dim_raw = int(cond.shape[1])
    # Auto-PCA high-dim conditioning (e.g. the 192-d latent) to keep k-NN local.
    n_pca = args.cond_pca or (32 if cond.shape[1] > 48 else 0)
    if n_pca:
        cond = pca_reduce(cond, n_pca)

    stats = analyze(cond, target, args.n_anchors, args.k, rng)
    result = {'tag': args.tag, 'dataset': args.dataset, 'mode': args.mode,
              'n_pairs': int(cond.shape[0]), 'cond_dim_raw': cond_dim_raw,
              'cond_dim_used': int(cond.shape[1]), 'chunk': args.chunk,
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
