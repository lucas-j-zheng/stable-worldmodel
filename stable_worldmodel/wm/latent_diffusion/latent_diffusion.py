import copy
import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from stable_worldmodel.wm.utils import load_pretrained


class LatentDiffusionDynamics(nn.Module):
    """Diffusion dynamics model over frozen LEWM latents.

    The denoiser is trained on a fixed ``[history_size | horizon]`` token
    layout, so its positional/type embeddings are layout-specific. Those two
    values are stored on the model (and persisted in the checkpoint config) and
    enforced everywhere a trajectory is built, so a train/inference layout
    mismatch fails loudly instead of silently degrading sample quality.
    """

    def __init__(
        self,
        *,
        denoiser: nn.Module,
        lewm: nn.Module | None = None,
        lewm_checkpoint: str | None = None,
        freeze_lewm: bool = True,
        history_size: int | None = None,
        horizon: int | None = None,
        num_diffusion_steps: int = 100,
        schedule: str = 'cosine',
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        cosine_s: float = 8e-3,
        num_inference_steps: int = 10,
        eta: float = 0.0,
        k_samples: int = 1,
        clip_sample: float | None = 6.0,
    ):
        super().__init__()

        # ``lewm`` may arrive (a) already instantiated -- an inlined ``lewm``
        # config in a self-contained checkpoint (hydra builds it before calling
        # us) or a module handed in by a test, or (b) as a checkpoint name we
        # load here. ``lewm_config`` records the frozen encoder's architecture
        # so ``export_config`` can inline it, making diffusion checkpoints
        # self-contained (no dependency on the original LeWM file at load time).
        self.lewm_config = None
        if lewm is None:
            if lewm_checkpoint is None:
                raise ValueError(
                    'LatentDiffusionDynamics needs a frozen encoder: pass a '
                    '`lewm` module, a `lewm_checkpoint` to load it from, or a '
                    'checkpoint with an inlined `lewm` config. Got none.'
                )
            lewm = load_pretrained(lewm_checkpoint)
            self.lewm_config = self._resolve_lewm_config(lewm_checkpoint)

        self.lewm = lewm
        self.denoiser = denoiser
        self.lewm_checkpoint = lewm_checkpoint
        self.freeze_lewm = freeze_lewm
        self.history_size = (
            int(history_size) if history_size is not None else None
        )
        self.horizon = int(horizon) if horizon is not None else None
        self.num_diffusion_steps = int(num_diffusion_steps)
        self.num_inference_steps = int(num_inference_steps)
        self.schedule = schedule
        # DDIM stochasticity: eta=0 is deterministic DDIM (default). The frozen
        # SIGReg latent marginal is ~N(0, I), so the diffusion prior already
        # matches the data marginal -- no latent normalization is applied.
        self.eta = float(eta)
        # Number of independent dynamics samples averaged per candidate when
        # scoring action plans (D-MPC: average before ranking).
        self.k_samples = int(k_samples)
        # Static x0 thresholding (Nichol & Dhariwal). The cosine terminal
        # alpha_bar is ~1e-5, so predict_start divides by ~3e-3 at the first
        # DDIM step and amplifies eps error ~300x; without a clamp the error
        # compounds by sqrt(alpha_0/alpha_T) across the chain. SIGReg latents
        # are ~N(0, I) (|z| < ~5), so +/-6 is a safe envelope.
        self.clip_sample = float(clip_sample) if clip_sample else None

        # Fail fast at construction if the denoiser cannot fit the trained
        # trajectory layout, rather than truncating the sequence at runtime.
        if self.history_size is not None and self.horizon is not None:
            needed = self.history_size + self.horizon
            max_seq_len = getattr(denoiser, 'max_seq_len', None)
            if max_seq_len is not None and max_seq_len < needed:
                raise ValueError(
                    f'denoiser.max_seq_len ({max_seq_len}) < history_size + '
                    f'horizon ({self.history_size} + {self.horizon} = '
                    f'{needed}). Increase model.denoiser.max_seq_len.'
                )

        alphas_cumprod = self._build_alphas_cumprod(
            schedule,
            self.num_diffusion_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            cosine_s=cosine_s,
        )

        self.register_buffer('alphas_cumprod', alphas_cumprod, persistent=False)
        self.register_buffer(
            'sqrt_alphas_cumprod',
            torch.sqrt(alphas_cumprod),
            persistent=False,
        )
        self.register_buffer(
            'sqrt_one_minus_alphas_cumprod',
            torch.sqrt(1.0 - alphas_cumprod),
            persistent=False,
        )

        if freeze_lewm:
            for param in self.lewm.parameters():
                param.requires_grad_(False)
            self.lewm.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_lewm:
            self.lewm.eval()
        return self

    @property
    def latent_dim(self) -> int:
        return self.denoiser.latent_dim

    ##############################
    ##   layout / config guards ##
    ##############################

    def _check_history_size(self, history_size: int | None) -> int:
        """Resolve ``history_size`` against the trained value, asserting match."""
        if history_size is None:
            if self.history_size is None:
                raise ValueError(
                    'history_size was not provided and is not stored on the '
                    'model. Set model.history_size at construction.'
                )
            return self.history_size
        history_size = int(history_size)
        if self.history_size is not None and history_size != self.history_size:
            raise ValueError(
                f'history_size mismatch: got {history_size}, model was trained '
                f'with {self.history_size}. The denoiser layout is fixed; match '
                'the trained value (set plan_config.history_len accordingly).'
            )
        return history_size

    def _check_horizon(self, horizon: int | None) -> int | None:
        """Resolve ``horizon`` against the trained value, asserting match."""
        if horizon is None:
            return self.horizon
        horizon = int(horizon)
        if self.horizon is not None and horizon != self.horizon:
            raise ValueError(
                f'horizon mismatch: got {horizon}, model was trained with '
                f'{self.horizon}. The denoiser layout is fixed; match the '
                'trained value (set plan_config.horizon accordingly).'
            )
        return horizon

    @staticmethod
    def _resolve_lewm_config(lewm_checkpoint: str) -> dict | None:
        """Read the frozen encoder's config so it can be inlined on save."""
        try:
            from stable_worldmodel.data import get_cache_dir
            from stable_worldmodel.wm.utils import _resolve

            cache_dir = get_cache_dir(None, sub_folder='checkpoints')
            _, config = _resolve(lewm_checkpoint, cache_dir)
            return config
        except Exception:  # pragma: no cover - best-effort capture
            return None

    def export_config(self, base_config):
        """Return a self-contained copy of ``base_config`` for saving.

        Inlines the frozen encoder architecture under a ``lewm`` key and clears
        ``lewm_checkpoint``, so the checkpoint reloads without the original LeWM
        file. The encoder weights travel inside the diffusion ``state_dict``.
        """
        from omegaconf import OmegaConf

        if OmegaConf.is_config(base_config):
            config = OmegaConf.to_container(base_config, resolve=True)
        else:
            config = copy.deepcopy(dict(base_config))

        if self.lewm_config is not None:
            config['lewm'] = copy.deepcopy(self.lewm_config)
            config['lewm_checkpoint'] = None
        return OmegaConf.create(config)

    @staticmethod
    def _build_alphas_cumprod(
        schedule: str,
        num_steps: int,
        *,
        beta_start: float,
        beta_end: float,
        cosine_s: float,
    ) -> torch.Tensor:
        """Cumulative-product alphas for the chosen noise schedule.

        ``cosine`` is the Nichol & Dhariwal (2021) schedule, whose terminal
        distribution is ~N(0, I) -- matching the frozen SIGReg latent marginal.
        ``linear`` keeps the original DDPM beta schedule for back-compat.
        """
        if schedule == 'linear':
            betas = torch.linspace(
                beta_start, beta_end, num_steps, dtype=torch.float32
            )
            return torch.cumprod(1.0 - betas, dim=0)

        if schedule == 'cosine':
            steps = torch.arange(num_steps + 1, dtype=torch.float32)
            f = torch.cos(
                ((steps / num_steps) + cosine_s)
                / (1.0 + cosine_s)
                * math.pi
                * 0.5
            ) ** 2
            alphas_cumprod = f / f[0]
            # Convert to per-step alphas and back so the discrete cumulative
            # product is internally consistent and strictly within (0, 1).
            betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = betas.clamp(1e-8, 0.999)
            return torch.cumprod(1.0 - betas, dim=0)

        raise ValueError(
            f"Unknown diffusion schedule '{schedule}'. Use 'cosine' or 'linear'."
        )

    def _extract(
        self, values: torch.Tensor, timesteps: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        out = values.gather(0, timesteps)
        return out.view(-1, *([1] * (target.ndim - 1))).to(
            device=target.device, dtype=target.dtype
        )

    def q_sample(
        self,
        x_start: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_alpha = self._extract(
            self.sqrt_alphas_cumprod, timesteps, x_start
        )
        sqrt_one_minus_alpha = self._extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, x_start
        )
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

    def predict_start_from_noise(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, timesteps, x_t)
        sqrt_one_minus_alpha = self._extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, x_t
        )
        return (x_t - sqrt_one_minus_alpha * noise) / sqrt_alpha.clamp_min(
            1e-6
        )

    @torch.no_grad()
    def encode_latents(self, info: dict) -> torch.Tensor:
        # Fast path: a precomputed ``latent`` column (e.g. from the offline
        # latent cache) lets training skip the frozen encoder entirely. Planning
        # still encodes raw observations online via the LeWM encoder below.
        if 'latent' in info and torch.is_tensor(info['latent']):
            return info['latent'].detach()
        encoded = self.lewm.encode(dict(info))
        return encoded['emb'].detach()

    @torch.no_grad()
    def encode_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return self.lewm.action_encoder(actions).detach()

    @staticmethod
    def _slice_pad_time(
        x: torch.Tensor, start: int, length: int
    ) -> torch.Tensor:
        """Take ``length`` steps from ``start`` along time.

        Pads the tail by repeating the last step only when the slice is short --
        i.e. an episode-boundary row with fewer than ``length`` actions. In
        normal training the dataset row already has ``history_size + horizon``
        steps, so no padding happens.
        """
        if length <= 0:
            return x[:, :0]

        start = max(start, 0)
        sliced = x[:, start : start + length]
        if sliced.shape[1] == length:
            return sliced

        pad_len = length - sliced.shape[1]
        if sliced.shape[1] == 0:
            pad_value = torch.zeros(
                x.shape[0],
                1,
                x.shape[-1],
                device=x.device,
                dtype=x.dtype,
            )
        else:
            pad_value = sliced[:, -1:]
        pad = pad_value.expand(-1, pad_len, -1)
        return torch.cat([sliced, pad], dim=1)

    @staticmethod
    def training_action_context(
        actions: torch.Tensor, history_size: int, horizon: int
    ) -> torch.Tensor:
        actions = torch.nan_to_num(actions, 0.0)
        return LatentDiffusionDynamics._slice_pad_time(
            actions, 0, history_size + horizon
        )

    def diffusion_loss(
        self,
        batch: dict,
        *,
        history_size: int | None = None,
        horizon: int | None = None,
    ) -> dict:
        history_size = self._check_history_size(history_size)
        horizon = self._check_horizon(horizon)
        emb = self.encode_latents(batch)
        horizon = emb.shape[1] - history_size if horizon is None else horizon

        if history_size <= 0:
            raise ValueError('history_size must be positive.')
        if horizon <= 0:
            raise ValueError('horizon must be positive.')
        if emb.shape[1] < history_size + horizon:
            raise ValueError(
                f'Need at least {history_size + horizon} latent steps, '
                f'got {emb.shape[1]}.'
            )

        dtype = next(self.denoiser.parameters()).dtype
        history = emb[:, :history_size].to(dtype=dtype)
        target = emb[:, history_size : history_size + horizon].to(dtype=dtype)

        action_context = self.training_action_context(
            batch['action'], history_size, horizon
        )
        action_emb = self.encode_actions(action_context).to(dtype=dtype)

        timesteps = torch.randint(
            0,
            self.num_diffusion_steps,
            (target.shape[0],),
            device=target.device,
        )
        noise = torch.randn_like(target)
        noisy_future = self.q_sample(target, timesteps, noise)
        pred_noise = self.denoiser(
            noisy_future, history, action_emb, timesteps
        )

        loss = F.mse_loss(pred_noise.float(), noise.float())
        return {
            'emb': emb,
            'history_emb': history,
            'target_emb': target,
            'pred_noise': pred_noise,
            'noise': noise,
            'diffusion_loss': loss,
        }

    def _inference_timesteps(self, device: torch.device) -> torch.Tensor:
        steps = torch.linspace(
            self.num_diffusion_steps - 1,
            0,
            self.num_inference_steps,
            device=device,
        )
        return steps.round().long()

    @torch.no_grad()
    def sample_future(
        self,
        history: torch.Tensor,
        action_context: torch.Tensor,
        *,
        horizon: int,
    ) -> torch.Tensor:
        self._check_history_size(history.shape[1])
        self._check_horizon(horizon)
        dtype = next(self.denoiser.parameters()).dtype
        history = history.to(dtype=dtype)
        action_emb = self.encode_actions(action_context).to(dtype=dtype)

        x = torch.randn(
            history.shape[0],
            horizon,
            history.shape[-1],
            device=history.device,
            dtype=dtype,
        )

        # DDIM sampling (Song et al., 2021). ``eta=0`` is deterministic; larger
        # eta interpolates toward ancestral (DDPM) sampling. Timesteps run from
        # most-noised to least-noised, so ``next_t`` is the lower-noise step.
        timesteps = self._inference_timesteps(history.device)
        for i, step in enumerate(timesteps):
            t = torch.full(
                (history.shape[0],), int(step.item()), device=history.device
            )
            pred_noise = self.denoiser(x, history, action_emb, t)
            pred_start = self.predict_start_from_noise(x, t, pred_noise)
            if self.clip_sample is not None:
                pred_start = pred_start.clamp(
                    -self.clip_sample, self.clip_sample
                )

            if i == len(timesteps) - 1:
                x = pred_start
                continue

            next_t = torch.full(
                (history.shape[0],),
                int(timesteps[i + 1].item()),
                device=history.device,
            )
            alpha_t = self._extract(self.alphas_cumprod, t, x)
            alpha_next = self._extract(self.alphas_cumprod, next_t, x)

            sigma = self.eta * torch.sqrt(
                (1.0 - alpha_next) / (1.0 - alpha_t).clamp_min(1e-12)
            ) * torch.sqrt((1.0 - alpha_t / alpha_next).clamp_min(0.0))
            direction = torch.sqrt(
                (1.0 - alpha_next - sigma**2).clamp_min(0.0)
            ) * pred_noise
            x = torch.sqrt(alpha_next) * pred_start + direction
            if self.eta > 0:
                x = x + sigma * torch.randn_like(x)

        return x

    def _fit_history(self, emb: torch.Tensor) -> torch.Tensor:
        """Normalize encoded history to the trained ``history_size``.

        Pads by repeating the earliest frame at episode start (the only
        legitimate short-history case, when fewer real frames exist) and keeps
        the most recent frames otherwise, so the denoiser always sees its
        trained layout. No-op when ``history_size`` is unknown (legacy ckpts).
        """
        if self.history_size is None:
            return emb
        t = emb.shape[1]
        if t == self.history_size:
            return emb
        if t > self.history_size:
            return emb[:, -self.history_size :]
        pad = emb[:, :1].expand(-1, self.history_size - t, -1)
        return torch.cat([pad, emb], dim=1)

    def _history_latents(
        self, info: dict, batch_size: int, num_samples: int
    ) -> torch.Tensor:
        if 'emb' not in info:
            # ``v[:, 0]`` drops the candidate-sample dim while KEEPING the
            # observation-time dim, so ``lewm.encode`` encodes the full history
            # stack (not just the first frame).
            init = {
                k: v[:, 0]
                for k, v in info.items()
                if torch.is_tensor(v) and k != 'action'
            }
            init_emb = self._fit_history(self.encode_latents(init))
            info['emb'] = (
                init_emb.unsqueeze(1)
                .expand(batch_size, num_samples, -1, -1)
                .detach()
            )
        return info['emb']

    @staticmethod
    def _history_actions(
        info: dict,
        batch_size: int,
        num_samples: int,
        history_size: int,
        action_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        actions = info.get('action')
        if torch.is_tensor(actions) and actions.shape[-1] == action_dim:
            # Match ``training_action_context``: reset-frame actions are NaN
            # (see the env wrapper), so zero them before conditioning.
            actions = torch.nan_to_num(actions, 0.0)
            if actions.ndim == 3:
                actions = actions.unsqueeze(1).expand(
                    batch_size, num_samples, -1, -1
                )
            return actions[:, :, -history_size:].to(device=device, dtype=dtype)

        return torch.zeros(
            batch_size,
            num_samples,
            history_size,
            action_dim,
            device=device,
            dtype=dtype,
        )

    @torch.no_grad()
    def rollout(
        self, info: dict, action_sequence: torch.Tensor
    ) -> dict:
        assert 'pixels' in info, 'pixels not in info dict'

        batch_size, num_samples, horizon, action_dim = action_sequence.shape
        self._check_horizon(horizon)
        history = self._history_latents(info, batch_size, num_samples)
        history_size = history.shape[2]

        # Training uses outgoing actions: action slot i maps latent i to i+1.
        # Online ``info['action']`` is incoming for each observed frame, so
        # shift history actions left and place the first planned action on the
        # last history token (the transition into the first predicted latent).
        history_actions = self._history_actions(
            info,
            batch_size,
            num_samples,
            history_size,
            action_dim,
            action_sequence.device,
            action_sequence.dtype,
        )
        action_context = torch.cat(
            [
                history_actions[:, :, 1:],
                action_sequence,
                # Fill the terminal conditioning slot; there is no in-horizon
                # successor transition for the final predicted latent.
                action_sequence[:, :, -1:],
            ],
            dim=2,
        )

        flat_history = rearrange(history, 'b s t d -> (b s) t d')
        flat_actions = rearrange(action_context, 'b s t a -> (b s) t a')

        future = self.sample_future(
            flat_history, flat_actions, horizon=horizon
        )
        predicted = torch.cat([flat_history, future], dim=1)
        info['predicted_emb'] = rearrange(
            predicted, '(b s) t d -> b s t d', b=batch_size, s=num_samples
        )
        return info

    def criterion(self, info: dict) -> torch.Tensor:
        pred_emb = info['predicted_emb']
        goal_emb = info['goal_emb']

        pred_final = pred_emb[:, :, -1]
        goal_final = (
            goal_emb[:, -1]
            .to(dtype=pred_final.dtype)
            .unsqueeze(1)
            .expand_as(pred_final)
        )
        return torch.linalg.vector_norm(
            pred_final - goal_final.detach(), ord=2, dim=-1
        )

    @torch.no_grad()
    def get_cost(
        self, info: dict, action_candidates: torch.Tensor
    ) -> torch.Tensor:
        assert 'goal' in info, 'goal not in info dict'

        if 'goal_emb' not in info:
            goal = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
            goal['pixels'] = goal['goal']

            for key in list(info.keys()):
                if key.startswith('goal_') and key in goal:
                    goal[key[len('goal_') :]] = goal.pop(key)

            goal.pop('action', None)
            info['goal_emb'] = self.encode_latents(goal)

        # Average the goal-distance score over K independent dynamics samples
        # per candidate (D-MPC: average *before* ranking), so the planner cannot
        # exploit whichever sample happened to hallucinate near the goal. The
        # history latents and goal latent are encoded once and cached on `info`;
        # each rollout re-draws fresh diffusion noise in `sample_future`.
        cost = None
        for _ in range(max(1, self.k_samples)):
            info = self.rollout(info, action_candidates)
            sample_cost = self.criterion(info)
            cost = sample_cost if cost is None else cost + sample_cost
        return cost / max(1, self.k_samples)


__all__ = ['LatentDiffusionDynamics']
