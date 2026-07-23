"""Is the latent-MPC planning cost block-blind? (D3b, diagnosis 2026-07-23)

The planner scores candidates by final-latent L2 to the goal latent
(``LatentDiffusionDynamics.criterion``). The Arc-4 capstone showed agent
position survives the encoder while contact/block structure doesn't — so this
cost may track the agent and largely ignore the block. If so, every latent-MPC
PushT eval in the program has been optimizing the wrong thing, which alone
explains the Q3 planability floors (and caps even the frozen baselines).

Method: for each episode, form (t, goal=t+K) pairs exactly as closed-loop eval
does (K = goal_offset_steps = 25). Encode the frames, then relate
d_lat = ||z_t - z_goal|| to the ground-truth components
d_agent = ||agent_t - agent_goal|| and d_block = ||blockpos_t - blockpos_goal||
(+ chordal block-angle distance). Report Spearman correlations and standardized
OLS coefficients of d_lat ~ d_agent + d_block. Coef ratio agent:block >> 1 =>
the cost is block-blind.

Run via SLURM (loads a real dataset + encoder). Example:
    python scripts/data/cost_goal_alignment.py \
        --dataset pusht_cslip05.lance --encoder lewm_pusht/weights.pt \
        --tag frozen_cslip05 --n-episodes 40
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def img_transform(img_size=224):
    import stable_pretraining as spt
    from torchvision.transforms import v2 as transforms

    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=img_size),
        ]
    )


def load_encoder(name, device):
    """A LeWM checkpoint or any dynamics checkpoint with a ``.lewm`` inside."""
    from stable_worldmodel.wm.utils import load_pretrained

    model = load_pretrained(name)
    enc = getattr(model, 'lewm', model)
    return enc.to(device).eval()


@torch.no_grad()
def encode_episode(enc, pixels, transform, device, chunk=64):
    """pixels: (T, H, W, C) uint8 -> (T, D) float32 latents."""
    frames = [transform(p) for p in pixels]  # each (C, H, W)
    x = torch.stack(frames).to(device)  # (T, C, H, W)
    out = []
    for i in range(0, x.shape[0], chunk):
        z = enc.encode({'pixels': x[i : i + chunk].unsqueeze(0)})['emb']
        out.append(z.squeeze(0).float().cpu())
    return torch.cat(out).numpy()


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / den) if den > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--encoder', required=True,
                    help='LeWM ckpt (lewm_pusht/weights.pt) or dynamics ckpt '
                         '(e2c_.../weights_epoch_5.pt — uses its .lewm)')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--n-episodes', type=int, default=40)
    ap.add_argument('--goal-offset', type=int, default=25,
                    help='steps ahead for the goal frame (eval uses 25)')
    ap.add_argument('--agent-dims', default='0,1')
    ap.add_argument('--block-dims', default='2,3')
    ap.add_argument('--angle-dim', type=int, default=4,
                    help='block angle index in state; -1 to disable')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    import stable_worldmodel as swm

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    enc = load_encoder(args.encoder, device)
    transform = img_transform()
    ds = swm.data.load_dataset(args.dataset)

    agent_ix = [int(i) for i in args.agent_dims.split(',')]
    block_ix = [int(i) for i in args.block_dims.split(',')]

    rng = np.random.default_rng(args.seed)
    eps = rng.permutation(len(ds.lengths))[: args.n_episodes]

    d_lat, d_agent, d_block = [], [], []
    state_dim = None
    for ep in eps:
        data = ds.load_episode(int(ep))
        T = int(ds.lengths[int(ep)])
        if T <= args.goal_offset + 1:
            continue
        s = np.asarray(data['state'], dtype=np.float64).reshape(T, -1)
        state_dim = s.shape[1]
        z = encode_episode(enc, np.asarray(data['pixels']), transform, device)
        K = args.goal_offset
        zt, zg = z[:-K], z[K:]
        st, sg = s[:-K], s[K:]
        d_lat.append(np.linalg.norm(zt - zg, axis=1))
        d_agent.append(np.linalg.norm(st[:, agent_ix] - sg[:, agent_ix], axis=1))
        db = np.linalg.norm(st[:, block_ix] - sg[:, block_ix], axis=1)
        if 0 <= args.angle_dim < s.shape[1]:
            # chordal angle distance, scaled to the block-pos length scale
            dang = np.abs(
                np.angle(np.exp(1j * (st[:, args.angle_dim] - sg[:, args.angle_dim])))
            )
            db = db + dang * (np.median(db[db > 0]) if (db > 0).any() else 1.0)
        d_block.append(db)

    d_lat = np.concatenate(d_lat)
    d_agent = np.concatenate(d_agent)
    d_block = np.concatenate(d_block)

    # standardized OLS: d_lat ~ d_agent + d_block
    X = np.stack(
        [
            (d_agent - d_agent.mean()) / (d_agent.std() + 1e-12),
            (d_block - d_block.mean()) / (d_block.std() + 1e-12),
            np.ones_like(d_agent),
        ],
        axis=1,
    )
    y = (d_lat - d_lat.mean()) / (d_lat.std() + 1e-12)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    r2 = 1.0 - ((y - pred) ** 2).mean()

    result = {
        'tag': args.tag,
        'dataset': args.dataset,
        'encoder': args.encoder,
        'n_pairs': int(d_lat.size),
        'state_dim': state_dim,
        'goal_offset': args.goal_offset,
        'spearman_lat_agent': round(spearman(d_lat, d_agent), 4),
        'spearman_lat_block': round(spearman(d_lat, d_block), 4),
        'coef_agent_std': round(float(coef[0]), 4),
        'coef_block_std': round(float(coef[1]), 4),
        'coef_ratio_agent_over_block': round(
            float(abs(coef[0]) / (abs(coef[1]) + 1e-12)), 3
        ),
        'ols_r2': round(float(r2), 4),
        'corr_agent_block': round(spearman(d_agent, d_block), 4),
    }
    if state_dim is not None and state_dim != 5:
        result['WARNING'] = (
            f'state_dim={state_dim} != 5; check --agent-dims/--block-dims'
        )
    print('[align] RESULT', json.dumps(result, indent=2))
    print(
        f"[align] >>> {args.tag}: coef agent={result['coef_agent_std']} "
        f"block={result['coef_block_std']} "
        f"ratio={result['coef_ratio_agent_over_block']} "
        f"(>>1 => planning cost is BLOCK-BLIND) <<<"
    )
    out = Path(f'logs/cost_goal_alignment_{args.tag}.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f'[align] wrote {out}')


if __name__ == '__main__':
    main()
