"""Tests for the D-MPC additions to the latent diffusion model.

Covers the cosine/DDIM schedule, the cached-latent fast path, and K-sample
averaging in get_cost. Uses a tiny stub LeWM so no vision encoder is built.
"""

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from stable_worldmodel.wm.latent_diffusion import (  # noqa: E402
    LatentDiffusionDynamics,
)
from stable_worldmodel.wm.latent_diffusion.module import (  # noqa: E402
    LatentTrajectoryDenoiser,
)


LATENT_DIM = 8
ACTION_RAW = 2
ACTION_EMB = 8
HISTORY = 3
HORIZON = 2


class StubLeWM(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_dim = LATENT_DIM
        self.enc = nn.Linear(LATENT_DIM, LATENT_DIM)
        self.action_encoder = nn.Linear(ACTION_RAW, ACTION_EMB)

    def encode(self, info):
        px = info["pixels"].float()
        b, t = px.shape[0], px.shape[1]
        return {"emb": self.enc(px.reshape(b, t, -1))}


def make_model(k_samples=1, schedule="cosine", eta=0.0):
    denoiser = LatentTrajectoryDenoiser(
        latent_dim=LATENT_DIM,
        action_dim=ACTION_EMB,
        hidden_dim=LATENT_DIM,
        max_seq_len=HISTORY + HORIZON,
        depth=1,
        heads=2,
        mlp_dim=16,
        dim_head=4,
    )
    return LatentDiffusionDynamics(
        denoiser=denoiser,
        lewm=StubLeWM(),
        history_size=HISTORY,
        horizon=HORIZON,
        num_diffusion_steps=20,
        num_inference_steps=4,
        schedule=schedule,
        eta=eta,
        k_samples=k_samples,
    )


# ----------------------------- schedule ----------------------------------- #


def test_cosine_schedule_is_monotone_and_bounded():
    acp = LatentDiffusionDynamics._build_alphas_cumprod(
        "cosine", 100, beta_start=1e-4, beta_end=2e-2, cosine_s=8e-3
    )
    assert acp.shape == (100,)
    assert torch.all(acp > 0) and torch.all(acp < 1)
    # Monotonically decreasing toward ~0 (terminal ~N(0, I)).
    assert torch.all(acp[1:] <= acp[:-1] + 1e-6)
    assert acp[0] > 0.9
    assert acp[-1] < 0.1


def test_linear_schedule_still_supported():
    acp = LatentDiffusionDynamics._build_alphas_cumprod(
        "linear", 50, beta_start=1e-4, beta_end=2e-2, cosine_s=8e-3
    )
    assert acp.shape == (50,)
    assert torch.all(acp[1:] <= acp[:-1] + 1e-6)


def test_unknown_schedule_raises():
    with pytest.raises(ValueError, match="schedule"):
        LatentDiffusionDynamics._build_alphas_cumprod(
            "quadratic", 10, beta_start=1e-4, beta_end=2e-2, cosine_s=8e-3
        )


def test_ddim_q_sample_predict_start_inverse_on_cosine():
    model = make_model(schedule="cosine")
    x0 = torch.randn(3, HORIZON, LATENT_DIM)
    noise = torch.randn_like(x0)
    t = torch.full((3,), 5, dtype=torch.long)
    recovered = model.predict_start_from_noise(
        model.q_sample(x0, t, noise), t, noise
    )
    torch.testing.assert_close(recovered, x0, atol=1e-4, rtol=1e-3)


# ----------------------------- cached latents ------------------------------ #


def test_encode_latents_uses_cached_latent_column():
    model = make_model()
    z = torch.randn(4, HISTORY + HORIZON, LATENT_DIM)
    # `pixels` present but should be ignored when a `latent` column exists.
    out = model.encode_latents({"latent": z, "pixels": torch.randn(4, 1, LATENT_DIM)})
    torch.testing.assert_close(out, z)


def test_diffusion_loss_on_latent_only_batch():
    model = make_model()
    total = HISTORY + HORIZON
    batch = {
        "latent": torch.randn(4, total, LATENT_DIM),
        "action": torch.randn(4, total, ACTION_RAW),
    }
    out = model.diffusion_loss(batch)  # no pixels, no encoder call
    assert out["diffusion_loss"].ndim == 0
    assert torch.isfinite(out["diffusion_loss"])


# ----------------------------- K-sample averaging -------------------------- #


def _make_info(b=2, s=4):
    return {
        "pixels": torch.randn(b, s, HISTORY, LATENT_DIM),
        "goal": torch.randn(b, s, HISTORY, LATENT_DIM),
    }, torch.randn(b, s, HORIZON, ACTION_RAW)


def test_get_cost_shape_and_finite_for_k_gt_1():
    model = make_model(k_samples=8).eval()
    info, actions = _make_info()
    cost = model.get_cost(info, actions)
    assert cost.shape == (2, 4)
    assert torch.isfinite(cost).all()


def test_criterion_uses_l2_not_squared_l2():
    model = make_model().eval()
    info = {
        "predicted_emb": torch.zeros(1, 2, HISTORY + HORIZON, LATENT_DIM),
        "goal_emb": torch.zeros(1, HISTORY, LATENT_DIM),
    }
    info["predicted_emb"][0, 0, -1, 0] = 3.0
    info["predicted_emb"][0, 0, -1, 1] = 4.0
    info["predicted_emb"][0, 1, -1, 0] = 6.0
    info["predicted_emb"][0, 1, -1, 1] = 8.0

    cost = model.criterion(info)

    torch.testing.assert_close(cost, torch.tensor([[5.0, 10.0]]))


def test_history_encoding_drops_raw_env_action(monkeypatch):
    model = make_model().eval()

    def encode_without_action(info):
        assert "action" not in info
        px = info["pixels"].float()
        b, t = px.shape[0], px.shape[1]
        return {"emb": model.lewm.enc(px.reshape(b, t, -1))}

    monkeypatch.setattr(model.lewm, "encode", encode_without_action)

    info, actions = _make_info()
    info["action"] = torch.randn(2, 4, HISTORY, 999)

    cost = model.get_cost(info, actions)

    assert cost.shape == (2, 4)


def test_rollout_aligns_first_planned_action_with_current_transition(monkeypatch):
    model = make_model().eval()
    captured = {}

    def capture_sample_future(history, action_context, *, horizon):
        captured["action_context"] = action_context.detach().clone()
        return torch.zeros(history.shape[0], horizon, LATENT_DIM)

    monkeypatch.setattr(model, "sample_future", capture_sample_future)

    info = {
        "pixels": torch.randn(1, 1, HISTORY, LATENT_DIM),
        # Online actions are incoming: a_i produced z_i. The first incoming
        # slot has no corresponding outgoing transition in the current history.
        "action": torch.tensor(
            [[[[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]]]]
        ),
    }
    planned = torch.tensor([[[[100.0, 100.0], [200.0, 200.0]]]])

    model.rollout(info, planned)

    expected = torch.tensor(
        [
            [
                [20.0, 20.0],
                [30.0, 30.0],
                [100.0, 100.0],
                [200.0, 200.0],
                [200.0, 200.0],
            ]
        ]
    )
    assert captured["action_context"].shape == (
        1,
        HISTORY + HORIZON,
        ACTION_RAW,
    )
    torch.testing.assert_close(captured["action_context"], expected)


def test_history_actions_passes_matching_blocks_and_zeros_nan():
    """When the policy supplies blocks whose last dim matches the planned action
    dim, they condition the denoiser (no zero fallback); reset-frame NaNs are
    zeroed exactly like training_action_context."""
    B, S, A = 2, 4, ACTION_RAW
    base = torch.arange(B * HISTORY * A, dtype=torch.float32).reshape(
        B, HISTORY, A
    )
    base[:, 0, :] = float("nan")  # reset-frame action
    info = {"action": base.clone()}

    out = LatentDiffusionDynamics._history_actions(
        info, B, S, HISTORY, A, torch.device("cpu"), torch.float32
    )

    assert out.shape == (B, S, HISTORY, A)
    assert torch.isfinite(out).all()  # NaN -> 0
    torch.testing.assert_close(out[:, 0, 0, :], torch.zeros(B, A))
    # Real actions are preserved and broadcast across the sample dim.
    torch.testing.assert_close(out[:, 0, -1, :], base[:, -1, :])


def test_history_actions_zero_fallback_on_dim_mismatch():
    """Raw single actions (last dim != block dim) fall back to zeros rather than
    silently feeding the wrong layout to the denoiser."""
    B, S, A = 2, 4, ACTION_RAW
    info = {"action": torch.randn(B, HISTORY, A)}  # single actions, dim A

    out = LatentDiffusionDynamics._history_actions(
        info, B, S, HISTORY, A * 5, torch.device("cpu"), torch.float32
    )

    assert out.shape == (B, S, HISTORY, A * 5)
    assert torch.count_nonzero(out) == 0


def test_k_averaging_of_identical_samples_equals_single(monkeypatch):
    """With a deterministic sampler, averaging K identical rollouts == K=1."""
    model = make_model(k_samples=1).eval()

    def deterministic_sample_future(history, action_context, *, horizon):
        # Independent of noise -> every rollout is identical.
        return history[:, :horizon] * 0.0 + history[:, :1].mean()

    monkeypatch.setattr(model, "sample_future", deterministic_sample_future)

    info1, actions = _make_info()
    info8 = {k: v.clone() for k, v in info1.items()}

    cost_k1 = model.get_cost(info1, actions)
    model.k_samples = 8
    cost_k8 = model.get_cost(info8, actions)

    torch.testing.assert_close(cost_k1, cost_k8)
