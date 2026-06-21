"""Denoiser for the latent Diffusion Policy.

Diffusion Policy (Chi et al. 2023) in the frozen-LeWM latent space: condition on
a history of latents + a goal latent, denoise a future ACTION chunk. Same
transformer backbone as the dynamics denoiser, but the diffused variable is the
action sequence (low-dim) and the latents are pure conditioning.

Token layout: [ history latents (H) | goal latent (1) | noisy actions (horizon) ].
Output: the denoised action tokens only, shape (B, horizon, action_dim).
"""

import torch
from torch import nn

from stable_worldmodel.wm.latent_diffusion.module import (
    SinusoidalTimestepEmbedding,
    TrajectoryBlock,
)


class ActionTrajectoryDenoiser(nn.Module):
    """Transformer denoiser predicting an action chunk from latent context.

    History latents and the goal latent are projected into the hidden space as
    conditioning tokens; the noisy action chunk is projected and denoised. A
    3-way type embedding (history / goal / action) tags the token roles.
    """

    def __init__(
        self,
        *,
        action_dim: int,
        latent_dim: int,
        goal_dim: int | None = None,
        hidden_dim: int = 192,
        max_seq_len: int = 12,
        depth: int = 6,
        heads: int = 8,
        mlp_dim: int = 1024,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.goal_dim = goal_dim or latent_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.goal_proj = nn.Linear(self.goal_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)

        self.time_mlp = nn.Sequential(
            SinusoidalTimestepEmbedding(hidden_dim),
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.SiLU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, hidden_dim) * 0.02
        )
        self.type_embedding = nn.Embedding(3, hidden_dim)  # 0=hist 1=goal 2=act
        self.dropout = nn.Dropout(emb_dropout)

        self.layers = nn.ModuleList(
            [
                TrajectoryBlock(
                    hidden_dim,
                    heads=heads,
                    dim_head=dim_head,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, action_dim)

    def forward(
        self,
        noisy_actions: torch.Tensor,   # (B, horizon, action_dim)
        history: torch.Tensor,         # (B, H, latent_dim)
        goal: torch.Tensor,            # (B, goal_dim) or (B, 1, goal_dim)
        timesteps: torch.Tensor,       # (B,)
    ) -> torch.Tensor:
        batch, hist_len, _ = history.shape
        horizon = noisy_actions.shape[1]
        if goal.ndim == 2:
            goal = goal.unsqueeze(1)
        seq_len = hist_len + 1 + horizon
        if seq_len > self.max_seq_len:
            raise ValueError(
                f'Sequence length {seq_len} exceeds max_seq_len '
                f'{self.max_seq_len}. Increase denoiser.max_seq_len.'
            )

        h = self.latent_proj(history)
        g = self.goal_proj(goal)
        a = self.action_proj(noisy_actions)
        x = torch.cat([h, g, a], dim=1)                     # (B, seq, hidden)
        x = x + self.pos_embedding[:, :seq_len].to(dtype=x.dtype)

        type_ids = torch.zeros(batch, seq_len, device=x.device, dtype=torch.long)
        type_ids[:, hist_len] = 1                           # goal token
        type_ids[:, hist_len + 1:] = 2                      # action tokens
        x = x + self.type_embedding(type_ids)

        time_emb = self.time_mlp(timesteps).to(dtype=x.dtype)
        x = self.dropout(x + time_emb.unsqueeze(1))

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.output_proj(x[:, hist_len + 1:])         # action tokens only


__all__ = ['ActionTrajectoryDenoiser']
