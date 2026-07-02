"""Rung-3 proposer bench: MSE vs generative heads on the goal-conditioned
K-step conditional p(s_{t+K} | s_t, goal).

The E6 screen (HJEPA_RESULTS.md R7/R8) showed this conditional is genuinely,
dose-dependently, hard-separately multimodal on multimodal-demonstrator TwoRoom
(mm0/dp05) and near-unimodal on p10. This bench trains small heads of each
family on the SAME data and evaluates them DISTRIBUTIONALLY per anchor against
the empirical conditional (k-NN in (s, goal) space on a held-out episode split),
giving the project's first det-vs-generative comparison on a conditional that
measurably passes the multimodality screen under matched conditioning.

Heads: mse | gauss (NLL) | mdn (k=5) | knn (retrieval sampler) | diff (DDPM MLP)

Per-anchor metrics (M model samples vs k empirical outcomes):
  energy   : energy distance between model samples and empirical outcomes
  precision: frac of model samples within 2*delta of some empirical outcome
             (delta = median NN spacing of the empirical set -- feasibility)
  coverage : among anchors whose empirical set is 2-mode (GMM BIC + sep>2sigma),
             frac of the 2 modes hit by at least one model sample (menu recall)
  meangap  : dist of the model's MEAN prediction to nearest empirical outcome,
             in delta units (the infeasible-average mechanism, direct)

Run via SLURM. Usage:
  python scripts/train/proposer_bench.py --dataset tworoom_mm0.lance --horizon 8
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import BallTree

GOAL_CANDS = ('goal_state', 'variation_target_position', 'goal',
              'target_position', 'goal_pos')


def build_pairs(ds, horizon, scol='state'):
    present = set(ds.column_names)
    gcol = next((c for c in GOAL_CANDS if c in present), None)
    if scol not in present or gcol is None:
        raise SystemExit(f'need {scol}+goal col; have {sorted(present)}')
    print(f'[bench] goal col = {gcol}')
    eps = []
    for ep in range(len(ds.lengths)):
        T = int(ds.lengths[ep])
        if T <= horizon:
            continue
        d = ds.load_episode(ep)
        s = np.asarray(d[scol], np.float32).reshape(T, -1)
        g = np.asarray(d[gcol], np.float32).reshape(T, -1)
        cond = np.concatenate([s[:T - horizon], g[:T - horizon]], 1)
        eps.append((cond, s[horizon:]))
    return eps


class MLP(nn.Module):
    def __init__(self, din, dout, hidden=256, depth=3):
        super().__init__()
        layers, d = [], din
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers += [nn.Linear(d, dout)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MDN(nn.Module):
    def __init__(self, din, dout, k=5):
        super().__init__()
        self.k, self.dout = k, dout
        self.trunk = MLP(din, k * (2 * dout + 1))

    def forward(self, x):
        o = self.trunk(x)
        k, d = self.k, self.dout
        logits = o[:, :k]
        mu = o[:, k:k + k * d].reshape(-1, k, d)
        logsig = o[:, k + k * d:].reshape(-1, k, d).clamp(-6, 2)
        return logits, mu, logsig

    def loss(self, x, y):
        logits, mu, logsig = self(x)
        z = (y[:, None] - mu) / logsig.exp()
        comp = (-0.5 * z.pow(2) - logsig
                - 0.5 * np.log(2 * np.pi)).sum(-1)          # (B, k)
        return -(torch.logsumexp(F.log_softmax(logits, -1) + comp, -1)).mean()

    @torch.no_grad()
    def sample(self, x, m):
        logits, mu, logsig = self(x)
        B = x.shape[0]
        out = []
        for _ in range(m):
            c = torch.multinomial(F.softmax(logits, -1), 1).squeeze(-1)
            idx = torch.arange(B, device=x.device)
            out.append(mu[idx, c] + logsig[idx, c].exp()
                       * torch.randn_like(mu[idx, c]))
        return torch.stack(out, 1)                            # (B, m, d)


class DDPM(nn.Module):
    def __init__(self, din_cond, dout, T=100):
        super().__init__()
        self.T, self.dout = T, dout
        self.net = MLP(dout + din_cond + 64, dout)
        t = torch.linspace(0, 1, T + 1)[1:]
        abar = torch.cos((t * 0.99 + 0.008) / 1.008 * np.pi / 2) ** 2
        self.register_buffer('abar', abar.clamp(1e-4, 0.9999))
        half = torch.exp(torch.arange(32) * (-np.log(10000.0) / 31))
        self.register_buffer('freq', half)

    def temb(self, t):                                        # (B,) int
        ang = t[:, None].float() * self.freq[None]
        return torch.cat([ang.sin(), ang.cos()], -1)          # (B, 64)

    def eps_hat(self, xt, t, cond):
        return self.net(torch.cat([xt, cond, self.temb(t)], -1))

    def loss(self, cond, y):
        B = y.shape[0]
        t = torch.randint(0, self.T, (B,), device=y.device)
        a = self.abar[t][:, None]
        eps = torch.randn_like(y)
        xt = a.sqrt() * y + (1 - a).sqrt() * eps
        return F.mse_loss(self.eps_hat(xt, t, cond), eps)

    @torch.no_grad()
    def sample(self, cond, m):
        B, d = cond.shape[0], self.dout
        c = cond.repeat_interleave(m, 0)
        x = torch.randn(B * m, d, device=cond.device)
        for ti in reversed(range(self.T)):
            t = torch.full((B * m,), ti, device=cond.device, dtype=torch.long)
            a = self.abar[ti]
            ap = self.abar[ti - 1] if ti > 0 else torch.tensor(1.0, device=cond.device)
            eps = self.eps_hat(x, t, c)
            x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-4, 4)
            mu = (ap.sqrt() * (1 - a / ap) / (1 - a)) * x0 \
                + ((a / ap).sqrt() * (1 - ap) / (1 - a)) * x
            if ti > 0:
                sig = ((1 - ap) / (1 - a) * (1 - a / ap)).sqrt()
                x = mu + sig * torch.randn_like(x)
            else:
                x = mu
        return x.reshape(B, m, d)


def train_head(name, Xtr, Ytr, dev, epochs=25, bs=512):
    din, dout = Xtr.shape[1], Ytr.shape[1]
    if name == 'mse':
        model, closs = MLP(din, dout).to(dev), \
            lambda m, x, y: F.mse_loss(m(x), y)
    elif name == 'gauss':
        net = MLP(din, 2 * dout).to(dev)

        def closs(m, x, y):
            o = m(x)
            mu, logsig = o[:, :dout], o[:, dout:].clamp(-6, 2)
            return (0.5 * ((y - mu) / logsig.exp()) ** 2 + logsig).sum(-1).mean()
        model = net
    elif name == 'mdn':
        model = MDN(din, dout).to(dev)
        closs = lambda m, x, y: m.loss(x, y)
    elif name == 'diff':
        model = DDPM(din, dout).to(dev)
        closs = lambda m, x, y: m.loss(x, y)
    else:
        raise ValueError(name)
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-5)
    X = torch.tensor(Xtr, device=dev)
    Y = torch.tensor(Ytr, device=dev)
    n = X.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = closs(model, X[idx], Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f'  [{name}] epoch {ep} loss {tot / n:.5f}')
    return model


@torch.no_grad()
def head_samples(name, model, Xva, m, dev, Ytr=None, tree=None, Ytr_all=None,
                 Xq=None):
    X = torch.tensor(Xva, device=dev)
    dout = Ytr_all.shape[1]
    if name == 'mse':
        mu = model(X)
        return mu[:, None].repeat(1, m, 1).cpu().numpy()
    if name == 'gauss':
        o = model(X)
        mu, sig = o[:, :dout], o[:, dout:].clamp(-6, 2).exp()
        s = mu[:, None] + sig[:, None] * torch.randn(
            X.shape[0], m, dout, device=dev)
        return s.cpu().numpy()
    if name == 'mdn':
        return model.sample(X, m).cpu().numpy()
    if name == 'diff':
        return model.sample(X, m).cpu().numpy()
    if name == 'knn':
        _, nbr = tree.query(Xq if Xq is not None else Xva, k=32)
        rng = np.random.default_rng(0)
        pick = rng.integers(0, 32, size=(Xva.shape[0], m))
        return Ytr_all[nbr[np.arange(len(Xva))[:, None], pick]]
    raise ValueError(name)


def evaluate(samples, Yemp_sets, mean_pred):
    """samples (N,m,d); Yemp_sets list of (k,d); mean_pred (N,d)."""
    en, pr, mg = [], [], []
    cov_hits, cov_tot = 0, 0
    for i, Ye in enumerate(Yemp_sets):
        S = samples[i]
        d_se = np.linalg.norm(S[:, None] - Ye[None], axis=-1)   # (m,k)
        d_ee = np.linalg.norm(Ye[:, None] - Ye[None], axis=-1)
        np.fill_diagonal(d_ee, np.inf)
        delta = np.median(d_ee.min(1)) + 1e-9
        d_ss = np.linalg.norm(S[:, None] - S[None], axis=-1)
        en.append(2 * d_se.mean() - d_ee[np.isfinite(d_ee)].mean()
                  - d_ss.mean())
        pr.append(float((d_se.min(1) < 2 * delta).mean()))
        mg.append(float(np.linalg.norm(mean_pred[i] - Ye, axis=-1).min()
                        / delta))
        # 2-mode coverage (in top-3 PC space of Ye when high-dim: a full-cov
        # GMM on k~64 samples in 192-d is singular; S is projected the same
        # way so distances are comparable)
        if Ye.shape[0] >= 16:
            mu0 = Ye.mean(0)
            Yp, Sp = Ye - mu0, S - mu0
            if Yp.shape[1] > 8:
                try:
                    _, _, vt = np.linalg.svd(Yp, full_matrices=False)
                    Yp, Sp = Yp @ vt[:3].T, Sp @ vt[:3].T
                except np.linalg.LinAlgError:
                    Yp, Sp = Yp[:, :3], Sp[:, :3]
            d_pp = np.linalg.norm(Yp[:, None] - Yp[None], axis=-1)
            np.fill_diagonal(d_pp, np.inf)
            delta_p = np.median(d_pp.min(1)) + 1e-9
            try:
                g1 = GaussianMixture(1, random_state=0).fit(Yp)
                g2 = GaussianMixture(2, random_state=0, n_init=2).fit(Yp)
                if g2.bic(Yp) < g1.bic(Yp):
                    sep = np.linalg.norm(g2.means_[0] - g2.means_[1])
                    sig = np.sqrt(g2.covariances_.reshape(2, -1).mean())
                    if sep > 2 * sig:
                        cov_tot += 2
                        for mu in g2.means_:
                            if np.linalg.norm(Sp - mu, axis=-1).min() \
                                    < 2 * delta_p:
                                cov_hits += 1
            except Exception:
                pass
    return {
        'energy_med': float(np.median(en)),
        'precision_mean': float(np.mean(pr)),
        'meangap_med': float(np.median(mg)),
        'mode_coverage': float(cov_hits / cov_tot) if cov_tot else None,
        'bimodal_anchors': cov_tot // 2,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--state-col', default='state',
                    help='column used as state/target (e.g. latent)')
    ap.add_argument('--horizon', type=int, default=8)
    ap.add_argument('--heads', nargs='+',
                    default=['mse', 'gauss', 'mdn', 'knn', 'diff'])
    ap.add_argument('--n-anchors', type=int, default=1000)
    ap.add_argument('--k-emp', type=int, default=64)
    ap.add_argument('--m-samples', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[bench] device={dev} dataset={args.dataset} K={args.horizon}')

    import stable_worldmodel as swm
    ds = swm.data.load_dataset(args.dataset)
    eps = build_pairs(ds, args.horizon, scol=args.state_col)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(eps))
    n_val = max(1, len(eps) // 10)
    va_idx = set(order[:n_val].tolist())
    Xtr = np.concatenate([eps[i][0] for i in range(len(eps))
                          if i not in va_idx])
    Ytr = np.concatenate([eps[i][1] for i in range(len(eps))
                          if i not in va_idx])
    Xva = np.concatenate([eps[i][0] for i in sorted(va_idx)])
    Yva = np.concatenate([eps[i][1] for i in sorted(va_idx)])
    print(f'[bench] train {Xtr.shape[0]} pairs / val {Xva.shape[0]} '
          f'({len(eps)} eps, {n_val} val eps)')

    xm, xs = Xtr.mean(0), Xtr.std(0) + 1e-8
    ym, ys = Ytr.mean(0), Ytr.std(0) + 1e-8
    Xtrn, Ytrn = (Xtr - xm) / xs, (Ytr - ym) / ys
    Xvan, Yvan = (Xva - xm) / xs, (Yva - ym) / ys

    # neighbor-QUERY space: PCA-reduce high-dim conditioning (192-d latent) so
    # k-NN neighborhoods stay local (mirrors the screen's auto-PCA). Models
    # still receive the full conditioning.
    if Xtrn.shape[1] > 48:
        _, s_, vt_ = np.linalg.svd(Xtrn - Xtrn.mean(0), full_matrices=False)
        P = vt_[:32].T
        Xtrq, Xvaq = Xtrn @ P, Xvan @ P
        print(f'[bench] query-space PCA {Xtrn.shape[1]}->32 '
              f'(var kept {float((s_[:32]**2).sum()/(s_**2).sum()):.3f})')
    else:
        Xtrq, Xvaq = Xtrn, Xvan

    # empirical conditional sets on the VAL split
    tree_va = BallTree(Xvaq)
    anchors = rng.choice(Xvan.shape[0],
                         min(args.n_anchors, Xvan.shape[0]), replace=False)
    _, nbr = tree_va.query(Xvaq[anchors], k=min(args.k_emp, Xvan.shape[0]))
    Yemp = [Yvan[row] for row in nbr]
    tree_tr = BallTree(Xtrq)

    results = {}
    for name in args.heads:
        print(f'[bench] === {name} ===')
        model = None
        if name not in ('knn',):
            model = train_head(name, Xtrn, Ytrn, dev, epochs=args.epochs)
        S = head_samples(name, model, Xvan[anchors], args.m_samples, dev,
                         tree=tree_tr, Ytr_all=Ytrn, Xq=Xvaq[anchors])
        mean_pred = S.mean(1)
        res = evaluate(S, Yemp, mean_pred)
        results[name] = res
        print(f'[bench] {name}: {json.dumps(res)}')

    suffix = '' if args.state_col == 'state' else f'_{args.state_col}'
    sfx_seed = '' if args.seed == 0 else f'_s{args.seed}'
    out = args.out or (f'logs/proposer_bench_{Path(args.dataset).stem}'
                       f'_k{args.horizon}{suffix}{sfx_seed}.json')
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(
        {'dataset': args.dataset, 'horizon': args.horizon,
         'n_train': int(Xtr.shape[0]), 'n_val': int(Xva.shape[0]),
         'heads': results}, indent=2))
    print(f'[bench] wrote {out}')


if __name__ == '__main__':
    main()
