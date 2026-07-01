"""Per-sample policy fit error — the P0a "smoothing test" (mechanism candidate a).

The 2026-06-30 attribution screen showed the conditional the policy actually
models is UNIMODAL (residual_bimodal ~0.006), so test-time mode sampling cannot
explain the +10 DP-over-TMSE win. Candidate (a): the MSE model is corrupted at
TRAINING time — finite capacity smooths across neighboring (history, goal) bins
whose demos took different doors, dragging predictions toward the between-modes
average even though each exact bin is unimodal.

Test: per-sample action-chunk error of a trained policy against the demo chunk,
in the model's NORMALIZED action space (comparable across datasets), on BOTH
splits (train fit error is the direct quantity for a training-time story).
Prediction if (a) is live: TMSE error on multimodal data (mm05) >> TMSE error
on unimodal data (p10), and the gap exceeds DP's.

Usage (hydra, mirrors train_diffusion_policy.py so the windowing/split match):
  python scripts/data/policy_fit_error.py data=tworoom_latent \
      data.dataset.name=tworoom_mm_latent2.lance seed=1 \
      +policy_ckpt=tmse_mm05_s1/weights_epoch_250.pt +tag=tmse_mm05_s1
`seed` MUST match the checkpoint's training seed so the val split is the one
actually held out.
"""

import json
import os
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf


@torch.no_grad()
def split_errors(model, loader, history_size, horizon, device):
    errs = []
    for batch in loader:
        emb = batch['latent'].to(device)
        actions = torch.nan_to_num(batch['action'], 0.0).to(device)
        history = emb[:, :history_size]
        goal = emb[:, history_size + horizon - 1]
        target_n = model.norm_action(actions[:, history_size:history_size + horizon])
        pred = model.sample_actions(history, goal)  # denormalized
        pred_n = model.norm_action(pred)
        per_sample = ((pred_n - target_n) ** 2).mean(dim=(1, 2))
        errs.append(per_sample.float().cpu().numpy())
    return np.concatenate(errs) if errs else np.array([])


@hydra.main(version_base=None, config_path='../train/config',
            config_name='diffusion_policy')
def run(cfg):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop('name')
    cache_dir = os.environ.get('LOCAL_DATASET_DIR', None)
    dataset = swm.data.load_dataset(dataset_name, transform=None,
                                    cache_dir=cache_dir, **dataset_cfg)
    dataset.transform = None

    # Same split as training (seed must match the checkpoint's training seed).
    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen)

    loader_cfg = {**cfg.loader}
    loader_cfg['shuffle'] = False
    loader_cfg['drop_last'] = False
    train_loader = torch.utils.data.DataLoader(train_set, **loader_cfg)
    val_loader = torch.utils.data.DataLoader(val_set, **loader_cfg)

    model = swm.wm.utils.load_pretrained(cfg.policy_ckpt)
    model = model.to(device).eval()
    model.requires_grad_(False)

    hs, hz = cfg.wm.history_size, cfg.wm.horizon
    out = {'tag': cfg.tag, 'policy_ckpt': cfg.policy_ckpt,
           'dataset': dataset_name, 'seed': int(cfg.seed)}
    arrays = {}
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        e = split_errors(model, loader, hs, hz, device)
        arrays[f'{split}_errors'] = e
        out[split] = {
            'n': int(e.size),
            'mean': float(e.mean()) if e.size else None,
            'median': float(np.median(e)) if e.size else None,
            'p90': float(np.quantile(e, 0.9)) if e.size else None,
            'max': float(e.max()) if e.size else None,
        }
        print(f"[fit] {cfg.tag} {split}: n={out[split]['n']} "
              f"mean={out[split]['mean']:.4f} median={out[split]['median']:.4f} "
              f"p90={out[split]['p90']:.4f}")

    # hydra moves cwd to its run dir; write next to the repo's logs/ like the
    # other diagnostics so the sbatch can collect the JSONs.
    logs = Path(hydra.utils.get_original_cwd()) / 'logs'
    logs.mkdir(exist_ok=True)
    with open(logs / f'fit_error_{cfg.tag}.json', 'w') as f:
        json.dump(out, f, indent=2)
    np.savez(logs / f'fit_error_{cfg.tag}.npz', **arrays)
    print(f'[fit] wrote logs/fit_error_{cfg.tag}.json + .npz')


if __name__ == '__main__':
    run()
