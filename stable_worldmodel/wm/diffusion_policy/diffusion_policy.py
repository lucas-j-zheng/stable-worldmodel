"""Latent Diffusion Policy: (latent history, goal latent) -> action chunk.

A conditional diffusion model that samples an ACTION sequence given the current
observation history and a goal, both encoded by the frozen LeWM encoder. Unlike
the diffusion *dynamics* model (which predicts the next latent and is mode-blind
to the policy), this models P(action_chunk | obs, goal) directly -- so where the
demonstrations are multimodal (multiple valid action sequences), it can commit to
one mode instead of mode-averaging like an MSE/Gaussian policy.

The Gaussian-diffusion math (cosine schedule, q_sample, eps/x0/v parametrization,
DDIM) mirrors LatentDiffusionDynamics; only the diffused variable (actions) and
the conditioning (history + goal, no action context) differ.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from stable_worldmodel.wm.utils import load_pretrained


class DiffusionPolicy(nn.Module):
    def __init__(
        self,
        *,
        denoiser: nn.Module,
        lewm: nn.Module | None = None,
        lewm_checkpoint: str | None = None,
        freeze_lewm: bool = True,
        action_dim: int,
        history_size: int,
        horizon: int,
        num_diffusion_steps: int = 100,
        schedule: str = 'cosine',
        cosine_s: float = 8e-3,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        num_inference_steps: int = 16,
        eta: float = 0.0,
        clip_sample: float | None = 3.0,
        prediction_type: str = 'v',
    ):
        super().__init__()
        if lewm is None:
            if lewm_checkpoint is None:
                raise ValueError('DiffusionPolicy needs `lewm` or `lewm_checkpoint`.')
            lewm = load_pretrained(lewm_checkpoint)
        self.lewm = lewm
        self.denoiser = denoiser
        self.lewm_checkpoint = lewm_checkpoint
        self.freeze_lewm = freeze_lewm
        self.action_dim = int(action_dim)
        self.history_size = int(history_size)
        self.horizon = int(horizon)
        self.num_diffusion_steps = int(num_diffusion_steps)
        self.num_inference_steps = int(num_inference_steps)
        self.eta = float(eta)
        self.clip_sample = float(clip_sample) if clip_sample else None
        if prediction_type not in ('eps', 'x0', 'v'):
            raise ValueError(f"bad prediction_type '{prediction_type}'")
        self.prediction_type = prediction_type

        alphas_cumprod = self._build_alphas_cumprod(
            schedule, self.num_diffusion_steps,
            beta_start=beta_start, beta_end=beta_end, cosine_s=cosine_s)
        self.register_buffer('alphas_cumprod', alphas_cumprod, persistent=False)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod), persistent=False)
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             torch.sqrt(1.0 - alphas_cumprod), persistent=False)
        # Action normalization (z-score) so the target matches the N(0,I) prior.
        # Set from data via set_action_stats; identity by default.
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

    # ---- Gaussian diffusion math (mirrors LatentDiffusionDynamics) ----
    @staticmethod
    def _build_alphas_cumprod(schedule, num_steps, *, beta_start, beta_end, cosine_s):
        if schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)
            return torch.cumprod(1.0 - betas, dim=0)
        if schedule == 'cosine':
            steps = torch.arange(num_steps + 1, dtype=torch.float32)
            f = torch.cos(((steps / num_steps) + cosine_s) / (1.0 + cosine_s)
                          * math.pi * 0.5) ** 2
            ac = f / f[0]
            betas = (1.0 - (ac[1:] / ac[:-1])).clamp(1e-8, 0.999)
            return torch.cumprod(1.0 - betas, dim=0)
        raise ValueError(f"Unknown schedule '{schedule}'")

    def _extract(self, values, timesteps, target):
        out = values.gather(0, timesteps)
        return out.view(-1, *([1] * (target.ndim - 1))).to(
            device=target.device, dtype=target.dtype)

    def q_sample(self, x_start, timesteps, noise):
        sa = self._extract(self.sqrt_alphas_cumprod, timesteps, x_start)
        soma = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start)
        return sa * x_start + soma * noise

    def target_from_noise_and_start(self, x_start, noise, timesteps):
        if self.prediction_type == 'eps':
            return noise
        if self.prediction_type == 'x0':
            return x_start
        sa = self._extract(self.sqrt_alphas_cumprod, timesteps, x_start)
        soma = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start)
        return sa * noise - soma * x_start

    def model_predictions(self, x_t, timesteps, model_out):
        sa = self._extract(self.sqrt_alphas_cumprod, timesteps, x_t)
        soma = self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_t)
        if self.prediction_type == 'eps':
            pred_noise = model_out
            pred_start = (x_t - soma * pred_noise) / sa.clamp_min(1e-6)
        elif self.prediction_type == 'x0':
            pred_start = model_out
            pred_noise = (x_t - sa * pred_start) / soma.clamp_min(1e-6)
        else:
            pred_start = sa * x_t - soma * model_out
            pred_noise = soma * x_t + sa * model_out
        if self.clip_sample is not None:
            pred_start = pred_start.clamp(-self.clip_sample, self.clip_sample)
            pred_noise = (x_t - sa * pred_start) / soma.clamp_min(1e-6)
        return pred_start, pred_noise

    @torch.no_grad()
    def encode_latents(self, info: dict) -> torch.Tensor:
        if 'latent' in info and torch.is_tensor(info['latent']):
            return info['latent'].detach()
        return self.lewm.encode(dict(info))['emb'].detach()

    # ---- training / inference ----
    def diffusion_loss(self, batch: dict, *, history_size=None, horizon=None) -> dict:
        history_size = history_size or self.history_size
        horizon = horizon or self.horizon
        emb = self.encode_latents(batch)                    # (B, T, latent)
        if emb.shape[1] < history_size + horizon:
            raise ValueError(f'need {history_size+horizon} latents, got {emb.shape[1]}')
        dtype = next(self.denoiser.parameters()).dtype
        history = emb[:, :history_size].to(dtype=dtype)
        goal = emb[:, history_size + horizon - 1].to(dtype=dtype)   # end-of-window
        actions = torch.nan_to_num(batch['action'], 0.0)
        target = self.norm_action(
            actions[:, history_size:history_size + horizon]).to(dtype=dtype)

        timesteps = torch.randint(0, self.num_diffusion_steps,
                                  (target.shape[0],), device=target.device)
        noise = torch.randn_like(target)
        noisy = self.q_sample(target, timesteps, noise)
        model_out = self.denoiser(noisy, history, goal, timesteps)
        reg_target = self.target_from_noise_and_start(target, noise, timesteps)
        loss = F.mse_loss(model_out.float(), reg_target.float())
        return {'diffusion_loss': loss, 'loss': loss}

    def _inference_timesteps(self, device):
        steps = torch.linspace(self.num_diffusion_steps - 1, 0,
                               self.num_inference_steps, device=device)
        return steps.round().long()

    @torch.no_grad()
    def sample_actions(self, history: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """DDIM sample a (B, horizon, action_dim) chunk; returns DE-normalized actions."""
        dtype = next(self.denoiser.parameters()).dtype
        history = history.to(dtype=dtype)
        goal = goal.to(dtype=dtype)
        x = torch.randn(history.shape[0], self.horizon, self.action_dim,
                        device=history.device, dtype=dtype)
        timesteps = self._inference_timesteps(history.device)
        for i, step in enumerate(timesteps):
            t = torch.full((history.shape[0],), int(step.item()), device=history.device)
            model_out = self.denoiser(x, history, goal, t)
            pred_start, pred_noise = self.model_predictions(x, t, model_out)
            if i == len(timesteps) - 1:
                x = pred_start
                continue
            next_t = torch.full((history.shape[0],), int(timesteps[i + 1].item()),
                                device=history.device)
            a_t = self._extract(self.alphas_cumprod, t, x)
            a_next = self._extract(self.alphas_cumprod, next_t, x)
            sigma = self.eta * torch.sqrt((1.0 - a_next) / (1.0 - a_t).clamp_min(1e-12)) \
                * torch.sqrt((1.0 - a_t / a_next).clamp_min(0.0))
            direction = torch.sqrt((1.0 - a_next - sigma ** 2).clamp_min(0.0)) * pred_noise
            x = torch.sqrt(a_next) * pred_start + direction
            if self.eta > 0:
                x = x + sigma * torch.randn_like(x)
        return self.denorm_action(x.float())


__all__ = ['DiffusionPolicy']
