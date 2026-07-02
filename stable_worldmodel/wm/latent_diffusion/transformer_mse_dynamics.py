"""Deterministic MSE control for the diffusion DYNAMICS model.

Same design as the policy-side TransformerMSEPolicy control (which caught the
architecture confound in the DP-vs-MLP comparison): reuse the EXACT
LatentTrajectoryDenoiser backbone, but train it to directly regress the future
latent window with MSE — no noise, fixed timestep 0. Then, on data with
genuinely multimodal dynamics (the slip env):

  diffusion > transformer-MSE  -> the win is the OBJECTIVE (mode-averaging of
                                  branched futures hurts planning)
  diffusion ~= transformer-MSE -> generative modeling buys nothing even on
                                  screened-multimodal dynamics.

Also fixes the old training-budget-parity confound: BOTH models are trained
post-hoc on the same cached latents with the same budget (the original D-MPC
baseline was the jointly-trained LeWM predictor).

Subclasses LatentDiffusionDynamics so the whole planning stack (rollout /
get_cost / criterion / CEM solver) is inherited; only the loss and the future
sampler change.
"""

import torch
import torch.nn.functional as F

from .latent_diffusion import LatentDiffusionDynamics


class TransformerMSEDynamics(LatentDiffusionDynamics):
    def _deterministic_forward(self, history, action_emb, horizon):
        dtype = next(self.denoiser.parameters()).dtype
        b = history.shape[0]
        zeros = torch.zeros(
            b, horizon, history.shape[-1],
            device=history.device, dtype=dtype,
        )
        t = torch.zeros(b, device=history.device, dtype=torch.long)
        return self.denoiser(zeros, history.to(dtype), action_emb.to(dtype), t)

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

        dtype = next(self.denoiser.parameters()).dtype
        history = emb[:, :history_size].to(dtype=dtype)
        target = emb[:, history_size : history_size + horizon].to(dtype=dtype)

        action_context = self.training_action_context(
            batch['action'], history_size, horizon
        )
        action_emb = self.encode_actions(action_context).to(dtype=dtype)

        pred = self._deterministic_forward(history, action_emb, horizon)
        loss = F.mse_loss(pred.float(), target.float())
        return {
            'emb': emb,
            'history_emb': history,
            'target_emb': target,
            'diffusion_loss': loss,
        }

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
        action_emb = self.encode_actions(action_context)
        return self._deterministic_forward(history, action_emb, horizon)


__all__ = ['TransformerMSEDynamics']
