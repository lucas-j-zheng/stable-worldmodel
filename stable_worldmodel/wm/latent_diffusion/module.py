import math

import torch
from torch import nn

from stable_worldmodel.wm.lewm.module import Attention, FeedForward


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal diffusion timestep embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim == 0:
            timesteps = timesteps[None]

        half = self.dim // 2
        if half == 0:
            return timesteps.float().unsqueeze(-1)

        denom = max(half - 1, 1)
        freqs = torch.exp(
            torch.arange(half, device=timesteps.device, dtype=torch.float32)
            * (-math.log(10000.0) / denom)
        )
        args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)

        if emb.shape[-1] < self.dim:
            emb = torch.nn.functional.pad(emb, (0, self.dim - emb.shape[-1]))

        return emb


class TrajectoryBlock(nn.Module):
    """Non-causal transformer block for latent trajectory denoising."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn = Attention(
            dim, heads=heads, dim_head=dim_head, dropout=dropout
        )
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), causal=False)
        x = x + self.mlp(self.norm2(x))
        return x


class LatentTrajectoryDenoiser(nn.Module):
    """Transformer denoiser for future LEWM latent chunks.

    The model receives clean history latents, noisy future latents, action
    embeddings for the whole token sequence, and the diffusion timestep. It
    predicts the diffusion noise on the future tokens only.
    """

    def __init__(
        self,
        *,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int,
        max_seq_len: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        self.latent_proj = (
            nn.Linear(latent_dim, hidden_dim)
            if latent_dim != hidden_dim
            else nn.Identity()
        )
        self.action_proj = (
            nn.Linear(action_dim, hidden_dim)
            if action_dim != hidden_dim
            else nn.Identity()
        )
        self.time_mlp = nn.Sequential(
            SinusoidalTimestepEmbedding(hidden_dim),
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.SiLU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, hidden_dim) * 0.02
        )
        self.type_embedding = nn.Embedding(2, hidden_dim)
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
        self.output_proj = (
            nn.Linear(hidden_dim, latent_dim)
            if hidden_dim != latent_dim
            else nn.Identity()
        )

    @staticmethod
    def _fit_sequence(x: torch.Tensor, seq_len: int) -> torch.Tensor:
        # The action context must cover the whole ``[history | future]``
        # trajectory, so its length always equals ``seq_len``. Mismatches mean a
        # layout bug upstream -- fail loudly rather than silently pad/truncate.
        if x.shape[1] != seq_len:
            raise ValueError(
                f'action embedding length {x.shape[1]} does not match token '
                f'sequence length {seq_len}; the action context must span the '
                'full history + future trajectory.'
            )
        return x

    def forward(
        self,
        noisy_future: torch.Tensor,
        history: torch.Tensor,
        action_emb: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        batch, history_len, _ = history.shape
        future_len = noisy_future.shape[1]
        seq_len = history_len + future_len

        if seq_len > self.max_seq_len:
            raise ValueError(
                f'Sequence length {seq_len} exceeds max_seq_len '
                f'{self.max_seq_len}. Increase model.denoiser.max_seq_len.'
            )

        tokens = torch.cat([history, noisy_future], dim=1)
        action_emb = self._fit_sequence(action_emb, seq_len)

        x = self.latent_proj(tokens)
        x = x + self.action_proj(action_emb)
        x = x + self.pos_embedding[:, :seq_len].to(dtype=x.dtype)

        type_ids = torch.zeros(
            batch, seq_len, device=x.device, dtype=torch.long
        )
        type_ids[:, history_len:] = 1
        x = x + self.type_embedding(type_ids)

        time_emb = self.time_mlp(timesteps).to(dtype=x.dtype)
        x = self.dropout(x + time_emb.unsqueeze(1))

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return self.output_proj(x[:, history_len:])


__all__ = [
    'LatentTrajectoryDenoiser',
    'SinusoidalTimestepEmbedding',
    'TrajectoryBlock',
]
