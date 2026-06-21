"""Train a latent Diffusion Policy (or the MSE baseline) on cached latents.

Mirrors scripts/train/latent_diffusion.py, with three differences for a POLICY
(action target instead of next-latent):
  1. The `action` column is NOT normalized by a data transform -- the model
     normalizes the action *target* internally (set_action_stats below), so the
     diffusion prior matches a ~N(0,I) action.
  2. Action mean/std are computed from the dataset and set on the model.
  3. The trainable submodule is configurable (`train_module`): `model.denoiser`
     for the diffusion policy, `model.net` for the MSE MLP baseline.
"""

import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

from stable_worldmodel.wm.utils import save_pretrained


class SaveCkptCallback(Callback):
    def __init__(self, run_name, cfg, epoch_interval: int = 5):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        if trainer.is_global_zero:
            epoch = trainer.current_epoch + 1
            if epoch % self.epoch_interval == 0 or epoch == trainer.max_epochs:
                config = (
                    pl_module.model.export_config(self.cfg.model)
                    if hasattr(pl_module.model, 'export_config')
                    else self.cfg.model
                )
                save_pretrained(pl_module.model, run_name=self.run_name,
                                config=config, filename=f'weights_epoch_{epoch}.pt')


def policy_forward(self, batch, stage, cfg):
    batch['action'] = torch.nan_to_num(batch['action'], 0.0)
    output = self.model.diffusion_loss(
        batch, history_size=cfg.wm.history_size, horizon=cfg.wm.horizon)
    output['loss'] = output['loss']
    losses = {f'{stage}/{k}': v.detach() for k, v in output.items() if 'loss' in k}
    self.log_dict(losses, on_step=True, sync_dist=True)
    return output


@hydra.main(version_base=None, config_path='./config', config_name='diffusion_policy')
def run(cfg):
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop('name')
    cache_dir = os.environ.get('LOCAL_DATASET_DIR', None)
    print(f'Loading dataset "{dataset_name}"')
    dataset = swm.data.load_dataset(dataset_name, transform=None,
                                    cache_dir=cache_dir, **dataset_cfg)

    keys_to_load = list(cfg.data.dataset.keys_to_load)
    transforms = []
    # NB: do NOT normalize `action` here -- the policy model normalizes the
    # action target internally. `latent` is already ~N(0,I) (SIGReg).
    dataset.transform = spt.data.transforms.Compose(*transforms) if transforms else None

    # Action stats for the model's internal normalization (NaN-safe over resets).
    acts = np.asarray(dataset.get_col_data('action'), dtype=np.float32)
    acts = acts.reshape(-1, acts.shape[-1])
    a_mean = np.nanmean(acts, axis=0)
    a_std = np.nanstd(acts, axis=0)
    print(f'[dp] action mean={a_mean} std={a_std}')

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen)
    train = torch.utils.data.DataLoader(train_set, **cfg.loader, generator=rnd_gen)
    val_cfg = {**cfg.loader}; val_cfg['shuffle'] = False; val_cfg['drop_last'] = False
    val = torch.utils.data.DataLoader(val_set, **val_cfg)

    model = hydra.utils.instantiate(cfg.model)
    model.set_action_stats(a_mean, a_std)

    total_steps = cfg.trainer.max_epochs * len(train)
    optimizers = {
        'model_opt': {
            'modules': cfg.get('train_module', 'model.denoiser'),
            'optimizer': dict(cfg.optimizer),
            'scheduler': {
                'type': 'LinearWarmupCosineAnnealingLR',
                'warmup_steps': max(1, int(0.01 * total_steps)),
                'max_steps': total_steps,
            },
            'interval': 'epoch',
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    module = spt.Module(model=model, forward=partial(policy_forward, cfg=cfg),
                        optim=optimizers)

    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'),
                   cfg.get('subdir') or '')
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    logger = None
    if cfg.get('wandb', {}).get('enabled', False):
        logger = WandbLogger(**cfg.wandb.config)

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[SaveCkptCallback(cfg.output_model_name, cfg,
                                    cfg.get('checkpoint_interval', 5))],
        num_sanity_val_steps=1, logger=logger, enable_checkpointing=True)

    manager = spt.Manager(trainer=trainer, module=module, data=data_module)
    manager()


if __name__ == '__main__':
    run()
