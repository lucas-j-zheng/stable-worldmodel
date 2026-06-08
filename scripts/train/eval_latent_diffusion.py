import json
import os
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf, open_dict
from stable_pretraining import data as dt
from torch.utils.data import DataLoader

from stable_worldmodel.data import column_normalizer as get_column_normalizer
from stable_worldmodel.wm.latent_diffusion import LatentDiffusionDynamics


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(
        **imagenet_stats, source=source, target=target
    )
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


def move_to_device(batch: dict, device: torch.device) -> dict:
    return {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


@torch.no_grad()
def encode_latents(lewm, batch: dict) -> torch.Tensor:
    return lewm.encode(dict(batch))['emb'].detach()


@torch.no_grad()
def lewm_open_loop(lewm, emb, actions, history_size: int, horizon: int):
    actions = torch.nan_to_num(actions, 0.0)
    action_context = LatentDiffusionDynamics.training_action_context(
        actions, history_size, horizon
    )
    act_emb = lewm.action_encoder(action_context)

    emb_list = list(emb[:, :history_size].unbind(dim=1))
    for t in range(horizon):
        lo = t
        emb_trunc = torch.stack(emb_list[lo:], dim=1)
        act_trunc = act_emb[:, lo : history_size + t]
        emb_list.append(lewm.predict(emb_trunc, act_trunc)[:, -1])

    return torch.stack(emb_list[history_size:], dim=1)


@torch.no_grad()
def diffusion_open_loop(model, emb, actions, history_size, horizon, n_samples):
    actions = torch.nan_to_num(actions, 0.0)
    history = emb[:, :history_size]
    action_context = model.training_action_context(
        actions, history_size, horizon
    )

    if n_samples <= 1:
        return model.sample_future(history, action_context, horizon=horizon)[
            None
        ]

    history = history.repeat_interleave(n_samples, dim=0)
    action_context = action_context.repeat_interleave(n_samples, dim=0)
    future = model.sample_future(history, action_context, horizon=horizon)
    future = future.view(-1, n_samples, horizon, future.shape[-1])
    return future.permute(1, 0, 2, 3).contiguous()


def step_mse(pred, target):
    return (pred.float() - target.float()).square().mean(dim=-1)


def summarize(metrics: dict[str, list[np.ndarray]]) -> dict:
    summary = {}
    for key, values in metrics.items():
        arr = np.concatenate(values, axis=0)
        summary[key] = float(arr.mean())
        if arr.ndim > 1:
            summary[f'{key}_per_step'] = arr.mean(axis=0).tolist()
            summary[f'{key}_final'] = float(arr[:, -1].mean())
    return summary


def default_output_path(diffusion_checkpoint: str) -> Path:
    ckpt_root = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'))
    ckpt_path = Path(diffusion_checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = ckpt_root / ckpt_path

    out_dir = ckpt_path.parent if ckpt_path.suffix == '.pt' else ckpt_path
    return out_dir / 'latent_diffusion_stage0.json'


@hydra.main(
    version_base=None,
    config_path='./config',
    config_name='latent_diffusion_eval',
)
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop('name')
    cache_dir = os.environ.get('LOCAL_DATASET_DIR', None)
    print(
        f'Loading dataset "{dataset_name}" from '
        f'{"local cache: " + cache_dir if cache_dir else "default location"}'
    )
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )

    transforms = [
        get_img_preprocessor(
            source='pixels', target='pixels', img_size=cfg.img_size
        )
    ]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith('pixels'):
                continue
            transforms.append(get_column_normalizer(dataset, col, col))

    dataset.transform = spt.data.transforms.Compose(*transforms)

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    _, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )

    loader_cfg = {**cfg.loader}
    loader_cfg['shuffle'] = False
    loader_cfg['drop_last'] = False
    val = DataLoader(val_set, **loader_cfg)

    ##############################
    ##       models / eval      ##
    ##############################

    device = torch.device(cfg.device)
    diffusion = swm.wm.utils.load_pretrained(cfg.diffusion_checkpoint)
    diffusion = diffusion.to(device).eval()
    diffusion.requires_grad_(False)

    if cfg.get('lewm_checkpoint'):
        lewm = swm.wm.utils.load_pretrained(cfg.lewm_checkpoint)
    else:
        lewm = diffusion.lewm
    lewm = lewm.to(device).eval()
    lewm.requires_grad_(False)

    metrics = {
        'lewm_step_mse': [],
        'diffusion_mean_step_mse': [],
        'diffusion_best_step_mse': [],
        'diffusion_sample_variance': [],
    }

    for batch_idx, batch in enumerate(val):
        if cfg.num_batches is not None and batch_idx >= cfg.num_batches:
            break

        batch = move_to_device(batch, device)
        batch['action'] = torch.nan_to_num(batch['action'], 0.0)

        emb = encode_latents(lewm, batch)
        target = emb[
            :, cfg.wm.history_size : cfg.wm.history_size + cfg.wm.horizon
        ]

        lewm_pred = lewm_open_loop(
            lewm,
            emb,
            batch['action'],
            cfg.wm.history_size,
            cfg.wm.horizon,
        )
        diffusion_samples = diffusion_open_loop(
            diffusion,
            emb,
            batch['action'],
            cfg.wm.history_size,
            cfg.wm.horizon,
            cfg.num_diffusion_samples,
        )

        diffusion_mean = diffusion_samples.mean(dim=0)
        diffusion_sample_mse = step_mse(
            diffusion_samples, target.unsqueeze(0)
        )
        best_idx = diffusion_sample_mse.mean(dim=-1).argmin(dim=0)
        best = diffusion_sample_mse[
            best_idx,
            torch.arange(target.shape[0], device=target.device),
        ]

        metrics['lewm_step_mse'].append(
            step_mse(lewm_pred, target).cpu().numpy()
        )
        metrics['diffusion_mean_step_mse'].append(
            step_mse(diffusion_mean, target).cpu().numpy()
        )
        metrics['diffusion_best_step_mse'].append(best.cpu().numpy())
        metrics['diffusion_sample_variance'].append(
            diffusion_samples.float()
            .var(dim=0, unbiased=False)
            .mean(dim=-1)
            .cpu()
            .numpy()
        )

    summary = summarize(metrics)
    summary['num_diffusion_samples'] = cfg.num_diffusion_samples
    summary['num_batches'] = (
        len(metrics['lewm_step_mse'])
        if cfg.num_batches is None
        else min(cfg.num_batches, len(metrics['lewm_step_mse']))
    )
    summary['diffusion_checkpoint'] = cfg.diffusion_checkpoint
    summary['lewm_checkpoint'] = cfg.get('lewm_checkpoint')

    print(json.dumps(summary, indent=2))

    if cfg.output_path:
        output_path = Path(cfg.output_path)
    else:
        output_path = default_output_path(cfg.diffusion_checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w') as f:
        json.dump(
            {
                'config': OmegaConf.to_container(cfg, resolve=True),
                'summary': summary,
            },
            f,
            indent=2,
        )


if __name__ == '__main__':
    run()
