"""Transformer MSE baseline: SAME architecture as the diffusion policy, but a
deterministic MSE-regression objective instead of denoising.

This is the control the original DiffusionPolicy-vs-MLP comparison was missing.
That comparison confounded TWO differences at once: architecture (6-layer
transformer denoiser vs 3-layer MLP) AND objective (diffusion sampling vs MSE
mode-averaging). To attribute any win to *diffusion / multimodality* rather than
to *the transformer*, hold architecture fixed: reuse the exact
ActionTrajectoryDenoiser backbone but train it to directly regress the action
chunk with MSE (no noise, fixed timestep 0). Then:
  diffusion > transformer-MSE  -> the win is the OBJECTIVE (mode-averaging hurts)
  diffusion ~= transformer-MSE -> the earlier win was ARCHITECTURE, not diffusion.

Mirrors the DiffusionPolicy/MLPPolicy interface (encode_latents / diffusion_loss
/ sample_actions) so it drops into the same training + eval harness.
"""

import torch
import torch.nn.functional as F
from torch import nn

from stable_worldmodel.wm.utils import load_pretrained


class TransformerMSEPolicy(nn.Module):
    def __init__(
        self,
        *,
        denoiser: nn.Module,          # an ActionTrajectoryDenoiser (same arch)
        lewm: nn.Module | None = None,
        lewm_checkpoint: str | None = None,
        freeze_lewm: bool = True,
        action_dim: int,
        history_size: int,
        horizon: int,
    ):
        super().__init__()
        if lewm is None:
            if lewm_checkpoint is None:
                raise ValueError('TransformerMSEPolicy needs `lewm`/`lewm_checkpoint`.')
            lewm = load_pretrained(lewm_checkpoint)
        self.lewm = lewm
        self.denoiser = denoiser
        self.lewm_checkpoint = lewm_checkpoint
        self.freeze_lewm = freeze_lewm
        self.action_dim = int(action_dim)
        self.history_size = int(history_size)
        self.horizon = int(horizon)
        self.register_buffer('action_mean', torch.zeros(self.action_dim))
        self.register_buffer('action_std', torch.ones(self.action_dim))
        if freeze_lewm:
            for p in self.lewm.parameters():
                p.requires_grad_(False)
            self.lewm.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_lewm:
            self.lewm.eval()
        return self

    def set_action_stats(self, mean, std):
        self.action_mean.copy_(torch.as_tensor(mean, dtype=self.action_mean.dtype))
        self.action_std.copy_(torch.as_tensor(std, dtype=self.action_std.dtype).clamp_min(1e-6))

    def norm_action(self, a):
        return (a - self.action_mean) / self.action_std

    def denorm_action(self, a):
        return a * self.action_std + self.action_mean

    @torch.no_grad()
    def encode_latents(self, info: dict) -> torch.Tensor:
        if 'latent' in info and torch.is_tensor(info['latent']):
            return info['latent'].detach()
        return self.lewm.encode(dict(info))['emb'].detach()

    def _predict(self, history, goal):
        # Deterministic forward through the denoiser: zero "noisy actions",
        # fixed timestep 0. The transformer maps (history, goal) -> action chunk.
        dtype = next(self.denoiser.parameters()).dtype
        b = history.shape[0]
        zeros = torch.zeros(b, self.horizon, self.action_dim,
                            device=history.device, dtype=dtype)
        t = torch.zeros(b, device=history.device, dtype=torch.long)
        return self.denoiser(zeros, history.to(dtype), goal.to(dtype), t)

    def diffusion_loss(self, batch: dict, *, history_size=None, horizon=None) -> dict:
        history_size = history_size or self.history_size
        horizon = horizon or self.horizon
        emb = self.encode_latents(batch)
        dtype = next(self.denoiser.parameters()).dtype
        history = emb[:, :history_size].to(dtype=dtype)
        goal = emb[:, history_size + horizon - 1].to(dtype=dtype)
        actions = torch.nan_to_num(batch['action'], 0.0)
        target = self.norm_action(actions[:, history_size:history_size + horizon]).to(dtype=dtype)
        pred = self._predict(history, goal)
        loss = F.mse_loss(pred.float(), target.float())
        return {'mse_loss': loss, 'loss': loss}

    @torch.no_grad()
    def sample_actions(self, history: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.denorm_action(self._predict(history, goal).float())


__all__ = ['TransformerMSEPolicy']
