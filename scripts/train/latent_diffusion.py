import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict
from stable_pretraining import data as dt

from stable_worldmodel.data import column_normalizer as get_column_normalizer
from stable_worldmodel.wm.loss import SIGReg
from stable_worldmodel.wm.utils import save_pretrained


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(
        **imagenet_stats, source=source, target=target
    )
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            epoch = trainer.current_epoch + 1
            if epoch % self.epoch_interval == 0 or epoch == trainer.max_epochs:
                self._save(pl_module.model, epoch)

    def _save(self, model, epoch):
        # Inline the frozen LeWM architecture into the saved config so the
        # checkpoint reloads without the original lewm_checkpoint file.
        config = (
            model.export_config(self.cfg.model)
            if hasattr(model, 'export_config')
            else self.cfg.model
        )
        save_pretrained(
            model,
            run_name=self.run_name,
            config=config,
            filename=f'weights_epoch_{epoch}.pt',
        )


def latent_diffusion_forward(self, batch, stage, cfg):
    """Encode with frozen LEWM and train diffusion noise prediction."""
    batch['action'] = torch.nan_to_num(batch['action'], 0.0)

    output = self.model.diffusion_loss(
        batch,
        history_size=cfg.wm.history_size,
        horizon=cfg.wm.horizon,
    )
    output['loss'] = output['diffusion_loss']

    # E2E + SIGReg (ARC 5b): with a trainable encoder, the bare dynamics loss
    # COLLAPSES the representation (measured: frozen global_std 0.97 -> e2e-diff
    # 0.2, e2e-tmse 0.04 — planner floors). Retain the SIGReg anti-collapse
    # regularizer as an auxiliary loss, exactly as in lewm.py pretraining.
    if getattr(self, 'sigreg', None) is not None:
        output['sigreg_loss'] = self.sigreg(output['emb'].transpose(0, 1))
        output['loss'] = (
            output['loss'] + cfg.sigreg_weight * output['sigreg_loss']
        )

    losses = {
        f'{stage}/{k}': v.detach() for k, v in output.items() if 'loss' in k
    }
    self.log_dict(losses, on_step=True, sync_dist=True)
    return output


@hydra.main(
    version_base=None, config_path='./config', config_name='latent_diffusion'
)
def run(cfg):
    # Full run seeding — cfg.seed alone only seeded the train/val split (same
    # bug the policy script had; audit 2026-06-29 / P0a).
    pl.seed_everything(cfg.seed, workers=True)

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

    keys_to_load = list(cfg.data.dataset.keys_to_load)

    transforms = []
    # Only preprocess pixels when training from raw observations. When training
    # on cached latents (the D-MPC pipeline) there is no `pixels` column.
    if any(k.startswith('pixels') for k in keys_to_load):
        transforms.append(
            get_img_preprocessor(
                source='pixels', target='pixels', img_size=cfg.img_size
            )
        )

    with open_dict(cfg):
        for col in keys_to_load:
            # `latent` is already ~N(0, I) by SIGReg construction -- do not
            # normalize it; pixels are handled above.
            if col.startswith('pixels') or col == 'latent':
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

    dataset.transform = spt.data.transforms.Compose(*transforms)

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )

    train = torch.utils.data.DataLoader(
        train_set,
        **cfg.loader,
        generator=rnd_gen,
    )
    val_cfg = {**cfg.loader}
    val_cfg['shuffle'] = False
    val_cfg['drop_last'] = False
    val = torch.utils.data.DataLoader(val_set, **val_cfg)

    ##############################
    ##       model / optim      ##
    ##############################

    model = hydra.utils.instantiate(cfg.model)

    total_steps = cfg.trainer.max_epochs * len(train)
    optimizers = {
        'model_opt': {
            'modules': 'model.denoiser',
            'optimizer': dict(cfg.optimizer),
            'scheduler': {
                'type': 'LinearWarmupCosineAnnealingLR',
                'warmup_steps': max(1, int(0.01 * total_steps)),
                'max_steps': total_steps,
            },
            'interval': 'epoch',
        },
    }

    # E2E (ARC 5): when the encoder is trainable, give it its own optimizer at
    # a scaled-down LR (fine-tune the pretrained lewm, don't destroy it).
    # Requires training from pixels (data=pusht/tworoom, not *_latent — the
    # cached-latent fast path detaches and would leave these params gradless).
    if not cfg.model.get('freeze_lewm', True):
        enc_opt = dict(cfg.optimizer)
        enc_opt['lr'] = float(cfg.optimizer.lr) * cfg.get(
            'encoder_lr_scale', 0.1
        )
        # No weight decay on the pretrained encoder: AdamW decay is a direct,
        # gradient-independent shrink pressure on the latent scale — one more
        # collapse channel SIGReg would otherwise have to fight (ARC 5c).
        enc_opt['weight_decay'] = 0.0
        optimizers['encoder_opt'] = {
            'modules': 'model.lewm',
            'optimizer': enc_opt,
            'scheduler': {
                'type': 'LinearWarmupCosineAnnealingLR',
                'warmup_steps': max(1, int(0.01 * total_steps)),
                'max_steps': total_steps,
            },
            'interval': 'epoch',
        }
        print(
            f'[e2e] encoder UNFROZEN: model.lewm trains at lr={enc_opt["lr"]:.2e}'
        )

    data_module = spt.data.DataModule(train=train, val=val)
    module_kwargs = {}
    # ARC 5b: attach the SIGReg anti-collapse regularizer for e2e training
    # (same knots/num_proj as lewm pretraining; weight via +sigreg_weight=0.09).
    if float(cfg.get('sigreg_weight', 0.0)) > 0.0:
        module_kwargs['sigreg'] = SIGReg(knots=17, num_proj=1024)
        print(f"[e2e] SIGReg auxiliary loss ON (weight={cfg.sigreg_weight})")
    module = spt.Module(
        model=model,
        forward=partial(latent_diffusion_forward, cfg=cfg),
        optim=optimizers,
        **module_kwargs,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get('subdir') or ''
    run_dir = Path(
        swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id
    )

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name,
        cfg=cfg,
        epoch_interval=cfg.get('checkpoint_interval', 5),
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f'{cfg.output_model_name}_weights.ckpt'
    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()


if __name__ == '__main__':
    run()
