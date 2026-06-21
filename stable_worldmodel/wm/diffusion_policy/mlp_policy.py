"""MSE regression baseline for the diffusion policy comparison.

Same (latent history, goal latent) -> action chunk mapping, but a deterministic
MLP trained with MSE. This is the MODE-AVERAGING baseline: on multimodal demos
(e.g. two valid routes) it regresses to the mean action, which can be invalid
(e.g. straight into the wall between two doors). The whole point of the diffusion
policy is to beat this where the data is multimodal. Mirrors DiffusionPolicy's
interface (encode_latents / diffusion_loss / sample_actions) so it's a drop-in
for the same training + eval harness.
"""

import torch
import torch.nn.functional as F
from torch import nn

from stable_worldmodel.wm.utils import load_pretrained


class MLPPolicy(nn.Module):
    def __init__(
        self,
        *,
        lewm: nn.Module | None = None,
        lewm_checkpoint: str | None = None,
        freeze_lewm: bool = True,
        action_dim: int,
        latent_dim: int = 192,
        history_size: int = 3,
        horizon: int = 8,
        hidden_dim: int = 512,
        depth: int = 3,
    ):
        super().__init__()
        if lewm is None:
            if lewm_checkpoint is None:
                raise ValueError('MLPPolicy needs `lewm` or `lewm_checkpoint`.')
            lewm = load_pretrained(lewm_checkpoint)
        self.lewm = lewm
        self.lewm_checkpoint = lewm_checkpoint
        self.freeze_lewm = freeze_lewm
        self.action_dim = int(action_dim)
        self.history_size = int(history_size)
        self.horizon = int(horizon)

        in_dim = latent_dim * (history_size + 1)            # history + goal
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, action_dim * horizon)]
        self.net = nn.Sequential(*layers)

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

    def _forward(self, history, goal):
        dtype = next(self.net.parameters()).dtype
        x = torch.cat([history.reshape(history.shape[0], -1),
                       goal.reshape(goal.shape[0], -1)], dim=1).to(dtype=dtype)
        out = self.net(x)
        return out.view(-1, self.horizon, self.action_dim)

    def diffusion_loss(self, batch: dict, *, history_size=None, horizon=None) -> dict:
        history_size = history_size or self.history_size
        horizon = horizon or self.horizon
        emb = self.encode_latents(batch)
        dtype = next(self.net.parameters()).dtype
        history = emb[:, :history_size].to(dtype=dtype)
        goal = emb[:, history_size + horizon - 1].to(dtype=dtype)
        actions = torch.nan_to_num(batch['action'], 0.0)
        target = self.norm_action(actions[:, history_size:history_size + horizon]).to(dtype=dtype)
        pred = self._forward(history, goal)
        loss = F.mse_loss(pred.float(), target.float())
        return {'mse_loss': loss, 'loss': loss}

    @torch.no_grad()
    def sample_actions(self, history: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.denorm_action(self._forward(history, goal).float())


__all__ = ['MLPPolicy']
