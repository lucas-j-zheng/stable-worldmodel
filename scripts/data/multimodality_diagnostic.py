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


def _state_col(present):
    """The physical-state column, tolerating the LeRobot loader's `proprio`
    alias (observation.state -> proprio) when no explicit `state` exists."""
    for c in ('state', 'proprio'):
        if c in present:
            return c
    return None


def build_pairs(ds, max_frames, mode, rng, chunk=8, cond_from='latent',
                cond_cols_override=None, goal_offset=8, early_frac=0.5):
    """Collect within-episode (cond_t, target_t) pairs up to ~max_frames.

    dynamics     : cond = [state, action]_t,  target = latent_{t+1}.
    policy       : cond = [state]_t,          target = action_t.
    policy_chunk : cond = obs_t, target = flattened action chunk [a_t..a_{t+H-1}].
                   The faithful Diffusion-Policy setting: multimodality lives in
                   obs-conditioned action *sequences*, not single-step actions.
                   cond_from='latent' (partial obs the policy sees) or 'state'
                   (ground-truth physical state -- ENCODER-FREE, the cleanest
                   cross-dataset test with no domain-gap confound).
    policy_goal  : cond = [state_t, state_{t+goal_offset}] (current + SYNTHESIZED
                   goal, exactly as closed-loop eval forms the goal), target =
                   action chunk [a_t..a_{t+chunk-1}]. This is the literal dose of
                   POLICY multimodality the diffusion-vs-MSE comparison hinges on:
                   for the same (state, goal) did the expert take distinct action
                   sequences? Encoder-free, fully deconfounded by the goal.
    `state` falls back to the LeRobot `proprio` alias when absent.
    """
    present = set(ds.column_names)
    scol = _state_col(present)

    if mode == 'policy_target':
        # Goal = the episode's FINAL state (the destination), which is the SAME
        # whichever door the agent took -> does NOT leak the route. So for the
        # same (state_t, target) a random-door expert has 2 action modes and a
        # greedy expert has 1. This is the CORRECT dose measurement (policy_goal
        # leaked the door by conditioning on the realized near-future position).
        # Restrict to the early `early_frac` of each episode (the room-A decision
        # phase, where the door choice lives); later travel is unimodal and only
        # dilutes the signal.
        if not scol or 'action' not in present:
            raise SystemExit(f'policy_target needs state+action; have {sorted(present)}')
        print(f'[mm] mode=policy_target cond=[{scol}_t, {scol}_final] '
              f'target=action_chunk(H={chunk}) early_frac={early_frac}')
        ep_order = rng.permutation(len(ds.lengths))
        conds, targs = [], []
        n = 0
        for ep in ep_order:
            ep = int(ep)
            T = int(ds.lengths[ep])
            if T <= chunk + 1:
                continue
            data = ds.load_episode(ep)
            s = np.asarray(data[scol], dtype=np.float32).reshape(T, -1)
            a = np.asarray(data['action'], dtype=np.float32).reshape(T, -1)
            goal = s[-1]                                   # destination
            t_end = max(1, int((T - chunk) * early_frac))
            for t in range(0, t_end):
                conds.append(np.concatenate([s[t], goal]))
                targs.append(a[t:t + chunk].reshape(-1))
            n += t_end
            if n >= max_frames:
                break
        cond = np.asarray(conds, dtype=np.float32)
        tgt = np.asarray(targs, dtype=np.float32)
        print(f'[mm] {cond.shape[0]} pairs, cond_dim={cond.shape[1]} '
              f'(state+target), target_dim={tgt.shape[1]} ({chunk}x action)')
        return cond, tgt

    if mode == 'policy_goal':
        if not scol or 'action' not in present:
            raise SystemExit(f'policy_goal needs state+action; have {sorted(present)}')
        print(f'[mm] mode=policy_goal cond=[{scol}_t, {scol}_t+{goal_offset}] '
              f'target=action_chunk(H={chunk})')
        ep_order = rng.permutation(len(ds.lengths))
        conds, targs = [], []
        n = 0
        for ep in ep_order:
            ep = int(ep)
            T = int(ds.lengths[ep])
            if T <= chunk + goal_offset:
                continue
            data = ds.load_episode(ep)
            s = np.asarray(data[scol], dtype=np.float32).reshape(T, -1)
            a = np.asarray(data['action'], dtype=np.float32).reshape(T, -1)
            for t in range(0, T - chunk - goal_offset):
                conds.append(np.concatenate([s[t], s[t + goal_offset]]))
                targs.append(a[t:t + chunk].reshape(-1))
            n += T - chunk - goal_offset
            if n >= max_frames:
                break
        cond = np.asarray(conds, dtype=np.float32)
        tgt = np.asarray(targs, dtype=np.float32)
        print(f'[mm] {cond.shape[0]} pairs, cond_dim={cond.shape[1]} '
              f'(state+goal), target_dim={tgt.shape[1]} ({chunk}x action)')
        return cond, tgt

    # Explicit conditioning columns (e.g. `state goal_state` = agent+target) let
    # the screen condition on the FULL deterministic structure so the detrend can
    # isolate the residual (e.g. door-choice) multimodality. Without strong
    # conditioning the screen just measures the marginal action distribution.
    override = list(cond_cols_override) if cond_cols_override else None
    if override:
        miss = [c for c in override if c not in present]
        if miss:
            raise SystemExit(f'--cond-cols {miss} absent; have {sorted(present)}')
    if mode == 'dynamics':
        cond_cols = override or [c for c in (scol, 'action') if c]
        if not override and 'action' not in cond_cols:
            raise SystemExit(f'dynamics needs action; have {sorted(present)}')
        target_col, shift = 'latent', True
    elif mode == 'policy':
        if not (override or scol) or 'action' not in present:
            raise SystemExit(f'policy needs state/proprio+action; have {sorted(present)}')
        cond_cols, target_col, shift = override or [scol], 'action', False
    elif mode == 'policy_chunk':
        src_cols = override or ([scol] if cond_from != 'latent' else ['latent'])
        if not all(src_cols) or 'action' not in present:
            raise SystemExit(f'policy_chunk needs {cond_from}+action; have {sorted(present)}')
    else:
        raise SystemExit(f'unknown mode {mode}')

    if mode == 'policy_chunk':
        src_cols = override or ([scol] if cond_from != 'latent' else ['latent'])
        print(f'[mm] mode=policy_chunk cond={src_cols} target=action_chunk(H={chunk})')
        ep_order = rng.permutation(len(ds.lengths))
        conds, targs = [], []
        n = 0
        for ep in ep_order:
            ep = int(ep)
            T = int(ds.lengths[ep])
            if T <= chunk:
                continue
            data = ds.load_episode(ep)
            c = np.concatenate(
                [np.asarray(data[col], dtype=np.float32).reshape(T, -1)
                 for col in src_cols], axis=1)
            a = np.asarray(data['action'], dtype=np.float32).reshape(T, -1)
            for t in range(0, T - chunk):
                conds.append(c[t])
                targs.append(a[t:t + chunk].reshape(-1))
            n += T - chunk
            if n >= max_frames:
                break
        cond = np.asarray(conds, dtype=np.float32)
        tgt = np.asarray(targs, dtype=np.float32)
        print(f'[mm] {cond.shape[0]} pairs, cond_dim={cond.shape[1]} ({src_cols}), '
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
    """k-NN-in-cond + local-linear detrend stats for P(target | cond).

    OVERFIT GUARD: the per-neighborhood detrend fits (cond_dim + 1) params from k
    neighbors. If k is not >> params, the fit overfits -> det_R2 inflated and the
    residual whitened toward Gaussian (residual_bimodal spuriously -> 0). We
    require k >= 10*(cond_dim + 1); auto-bump k if possible, and flag when even
    the data can't supply enough neighbors (result then under-powered, not wrong).
    """
    cmu, csd = cond.mean(0), cond.std(0) + 1e-8
    condn = (cond - cmu) / csd
    marginal_std = float(np.sqrt(target.var(0).mean()))

    n = condn.shape[0]
    params = condn.shape[1] + 1
    k_min = 10 * params
    k_eff = min(max(k, k_min), n - 1)
    overfit_risk = k_eff < 4 * params           # couldn't get a safe ratio
    if k_eff != k:
        print(f'[mm] overfit guard: cond_dim={condn.shape[1]} -> k {k}->{k_eff} '
              f'(need >= {k_min}; n={n})')
    if overfit_risk:
        print(f'[mm] WARNING: k_eff={k_eff} < 4*params={4*params}; detrend '
              f'under-powered, residual_bimodal unreliable for this run')
    k = k_eff

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
        'k_used': int(k),
        'params_per_fit': int(params),
        'overfit_risk': bool(overfit_risk),
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
    ap.add_argument('--mode',
                    choices=['dynamics', 'policy', 'policy_chunk', 'policy_goal',
                             'policy_target'],
                    default='dynamics')
    ap.add_argument('--chunk', type=int, default=8, help='action-chunk H (policy_chunk)')
    ap.add_argument('--goal-offset', type=int, default=8,
                    help='policy_goal: steps ahead for the synthesized goal state')
    ap.add_argument('--early-frac', type=float, default=0.5,
                    help='policy_target: fraction of each episode (from start) to use')
    ap.add_argument('--cond-from', choices=['latent', 'state'], default='latent',
                    help='policy_chunk conditioning source (state = encoder-free)')
    ap.add_argument('--cond-cols', nargs='+', default=None,
                    help='explicit conditioning columns (e.g. state goal_state)')
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
    cond, target = build_pairs(ds, args.max_frames, args.mode, rng, chunk=args.chunk,
                               cond_from=args.cond_from,
                               cond_cols_override=args.cond_cols,
                               goal_offset=args.goal_offset,
                               early_frac=args.early_frac)

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
