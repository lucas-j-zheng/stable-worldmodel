"""Tests for the latent diffusion forward-dynamics model.

These use a tiny stub LeWM (a couple of linear layers) so no real vision
encoder is built or downloaded. Pixels are passed already shaped
``(..., latent_dim)`` -- the stub treats them as pre-encoded features.
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
    """Minimal frozen-encoder stand-in for ``LeWM``.

    Must be importable by dotted path for the self-contained checkpoint test,
    so it lives at module scope.
    """

    def __init__(self, latent_dim=LATENT_DIM, action_raw=ACTION_RAW, action_emb=ACTION_EMB):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc = nn.Linear(latent_dim, latent_dim)
        self.action_encoder = nn.Linear(action_raw, action_emb)

    def encode(self, info):
        # pixels arrive as (..., T, latent_dim); treat them as pre-encoded.
        px = info["pixels"].float()
        b, t = px.shape[0], px.shape[1]
        feat = px.reshape(b, t, -1)
        return {"emb": self.enc(feat)}


def make_denoiser(max_seq_len=HISTORY + HORIZON, depth=1):
    return LatentTrajectoryDenoiser(
        latent_dim=LATENT_DIM,
        action_dim=ACTION_EMB,
        hidden_dim=LATENT_DIM,
        max_seq_len=max_seq_len,
        depth=depth,
        heads=2,
        mlp_dim=16,
        dim_head=4,
    )


def make_model(history=HISTORY, horizon=HORIZON, max_seq_len=HISTORY + HORIZON):
    return LatentDiffusionDynamics(
        denoiser=make_denoiser(max_seq_len=max_seq_len),
        lewm=StubLeWM(),
        history_size=history,
        horizon=horizon,
        num_diffusion_steps=20,
        num_inference_steps=4,
    )


def test_q_sample_predict_start_are_inverse():
    model = make_model()
    x0 = torch.randn(4, HORIZON, LATENT_DIM)
    noise = torch.randn_like(x0)
    t = torch.full((4,), 7, dtype=torch.long)

    x_t = model.q_sample(x0, t, noise)
    recovered = model.predict_start_from_noise(x_t, t, noise)

    torch.testing.assert_close(recovered, x0, atol=1e-5, rtol=1e-4)


def test_denoiser_output_shape():
    denoiser = make_denoiser()
    b = 3
    history = torch.randn(b, HISTORY, LATENT_DIM)
    noisy_future = torch.randn(b, HORIZON, LATENT_DIM)
    action_emb = torch.randn(b, HISTORY + HORIZON, ACTION_EMB)
    timesteps = torch.zeros(b, dtype=torch.long)

    out = denoiser(noisy_future, history, action_emb, timesteps)
    assert out.shape == (b, HORIZON, LATENT_DIM)


def test_denoiser_rejects_oversized_sequence():
    denoiser = make_denoiser(max_seq_len=HISTORY + HORIZON)
    b = 2
    history = torch.randn(b, HISTORY, LATENT_DIM)
    too_long_future = torch.randn(b, HORIZON + 2, LATENT_DIM)
    action_emb = torch.randn(b, HISTORY + HORIZON + 2, ACTION_EMB)
    timesteps = torch.zeros(b, dtype=torch.long)

    with pytest.raises(ValueError, match="max_seq_len"):
        denoiser(too_long_future, history, action_emb, timesteps)


def test_denoiser_rejects_misaligned_action_context():
    denoiser = make_denoiser()
    b = 2
    history = torch.randn(b, HISTORY, LATENT_DIM)
    noisy_future = torch.randn(b, HORIZON, LATENT_DIM)
    bad_action_emb = torch.randn(b, HISTORY, ACTION_EMB)  # missing future actions
    timesteps = torch.zeros(b, dtype=torch.long)

    with pytest.raises(ValueError, match="action embedding length"):
        denoiser(noisy_future, history, bad_action_emb, timesteps)


def test_construction_guards_max_seq_len():
    with pytest.raises(ValueError, match="max_seq_len"):
        LatentDiffusionDynamics(
            denoiser=make_denoiser(max_seq_len=HISTORY),  # < history + horizon
            lewm=StubLeWM(),
            history_size=HISTORY,
            horizon=HORIZON,
        )


def test_requires_an_encoder_source():
    with pytest.raises(ValueError, match="frozen encoder"):
        LatentDiffusionDynamics(denoiser=make_denoiser())


def test_sample_future_matches_trained_layout():
    model = make_model().eval()
    b = 5
    history = torch.randn(b, HISTORY, LATENT_DIM)
    action_context = torch.randn(b, HISTORY + HORIZON, ACTION_RAW)

    future = model.sample_future(history, action_context, horizon=HORIZON)
    assert future.shape == (b, HORIZON, LATENT_DIM)
    assert torch.isfinite(future).all()


def test_sample_future_rejects_layout_mismatch():
    model = make_model().eval()
    b = 2
    good_history = torch.randn(b, HISTORY, LATENT_DIM)
    bad_history = torch.randn(b, HISTORY + 1, LATENT_DIM)
    action_context = torch.randn(b, HISTORY + HORIZON, ACTION_RAW)

    with pytest.raises(ValueError, match="history_size mismatch"):
        model.sample_future(bad_history, action_context, horizon=HORIZON)

    with pytest.raises(ValueError, match="horizon mismatch"):
        model.sample_future(good_history, action_context, horizon=HORIZON + 3)


def test_diffusion_loss_uses_stored_layout():
    model = make_model()
    b, total = 4, HISTORY + HORIZON
    batch = {
        "pixels": torch.randn(b, total, LATENT_DIM),
        "action": torch.randn(b, total, ACTION_RAW),
    }

    out = model.diffusion_loss(batch)  # no explicit history/horizon
    assert out["diffusion_loss"].ndim == 0
    assert torch.isfinite(out["diffusion_loss"])

    with pytest.raises(ValueError, match="history_size mismatch"):
        model.diffusion_loss(batch, history_size=HISTORY + 1)


def test_get_cost_returns_per_candidate_costs():
    model = make_model().eval()
    b, s = 2, 4
    info = {
        "pixels": torch.randn(b, s, HISTORY, LATENT_DIM),
        "goal": torch.randn(b, s, HISTORY, LATENT_DIM),
    }
    actions = torch.randn(b, s, HORIZON, ACTION_RAW)

    cost = model.get_cost(info, actions)
    assert cost.shape == (b, s)
    assert torch.isfinite(cost).all()


def test_get_cost_rejects_horizon_mismatch():
    model = make_model().eval()
    b, s = 2, 3
    info = {
        "pixels": torch.randn(b, s, HISTORY, LATENT_DIM),
        "goal": torch.randn(b, s, HISTORY, LATENT_DIM),
    }
    actions = torch.randn(b, s, HORIZON + 2, ACTION_RAW)  # wrong horizon

    with pytest.raises(ValueError, match="horizon mismatch"):
        model.get_cost(info, actions)


def test_history_is_padded_to_trained_size_at_episode_start():
    model = make_model().eval()
    # Only one real history frame available (episode start) but trained on 3.
    single_frame = torch.randn(2, 1, LATENT_DIM)
    fitted = model._fit_history(single_frame)
    assert fitted.shape == (2, HISTORY, LATENT_DIM)
    # The earliest frame is repeated into the padded slots.
    torch.testing.assert_close(fitted[:, -1], single_frame[:, 0])


def _e2e_model(cls, freeze_lewm, stopgrad_target):
    return cls(
        denoiser=make_denoiser(),
        lewm=StubLeWM(),
        freeze_lewm=freeze_lewm,
        stopgrad_target=stopgrad_target,
        history_size=HISTORY,
        horizon=HORIZON,
        num_diffusion_steps=20,
        num_inference_steps=4,
    )


def _e2e_batch():
    return {
        "pixels": torch.randn(4, HISTORY + HORIZON, LATENT_DIM),
        "action": torch.randn(4, HISTORY + HORIZON, ACTION_RAW),
    }


def _encoder_grad_norm(model):
    out = model.diffusion_loss(_e2e_batch())
    model.zero_grad()
    out["diffusion_loss"].backward()
    grad = model.lewm.enc.weight.grad
    return (0.0 if grad is None else grad.norm().item()), out


def _mse_dynamics_cls():
    from stable_worldmodel.wm.latent_diffusion.transformer_mse_dynamics import (
        TransformerMSEDynamics,
    )

    return TransformerMSEDynamics


@pytest.mark.parametrize(
    "cls_factory",
    [lambda: LatentDiffusionDynamics, _mse_dynamics_cls],
    ids=["diffusion", "tmse"],
)
def test_e2e_stopgrad_target_grad_flow(cls_factory):
    """ARC 5c regression: the dynamics loss must not leak encoder gradients
    through the prediction-target branch (the measured ARC 5 collapse channel).

    Contract: frozen -> no encoder grads at all; e2e + stopgrad (default) ->
    grads via the history/conditioning branch only, target graph cut, the
    SIGReg path ('emb') keeps its graph; stopgrad_target=False reproduces the
    pre-fix behavior (target branch live) for old-run comparability.
    """
    cls = cls_factory()

    torch.manual_seed(0)
    grad, out = _encoder_grad_norm(_e2e_model(cls, True, True))
    assert grad == 0.0
    assert not out["target_emb"].requires_grad

    torch.manual_seed(0)
    grad, out = _encoder_grad_norm(_e2e_model(cls, False, True))
    assert grad > 0.0
    assert not out["target_emb"].requires_grad
    assert out["emb"].requires_grad

    torch.manual_seed(0)
    grad_old, out = _encoder_grad_norm(_e2e_model(cls, False, False))
    assert grad_old > 0.0
    assert out["target_emb"].requires_grad


def test_self_contained_checkpoint_round_trip(tmp_path, monkeypatch):
    """Saving inlines the LeWM config; reload works without the LeWM file."""
    pytest.importorskip("omegaconf")
    pytest.importorskip("hydra")
    from omegaconf import OmegaConf

    from stable_worldmodel.wm.utils import load_pretrained, save_pretrained

    # Route the cache dir (where load/save resolve checkpoints) to tmp_path.
    monkeypatch.setenv("SWM_CACHE", str(tmp_path))
    import stable_worldmodel.data as swm_data

    monkeypatch.setattr(
        swm_data, "get_cache_dir", lambda *a, **k: tmp_path / "checkpoints"
    )
    import stable_worldmodel.wm.utils as wm_utils

    monkeypatch.setattr(
        wm_utils, "get_cache_dir", lambda *a, **k: tmp_path / "checkpoints"
    )

    # 1) Save a stub LeWM as its own checkpoint (with an instantiable config).
    lewm = StubLeWM()
    lewm_cfg = OmegaConf.create(
        {
            "_target_": f"{__name__}.StubLeWM",
            "latent_dim": LATENT_DIM,
            "action_raw": ACTION_RAW,
            "action_emb": ACTION_EMB,
        }
    )
    save_pretrained(lewm, run_name="stub_lewm", config=lewm_cfg, filename="weights.pt")

    # 2) Build the diffusion model from that checkpoint (captures lewm config).
    base_cfg = OmegaConf.create(
        {
            "_target_": "stable_worldmodel.wm.latent_diffusion.LatentDiffusionDynamics",
            "lewm_checkpoint": "stub_lewm/weights.pt",
            "freeze_lewm": True,
            "history_size": HISTORY,
            "horizon": HORIZON,
            "num_diffusion_steps": 20,
            "num_inference_steps": 4,
            "denoiser": {
                "_target_": "stable_worldmodel.wm.latent_diffusion.module.LatentTrajectoryDenoiser",
                "latent_dim": LATENT_DIM,
                "action_dim": ACTION_EMB,
                "hidden_dim": LATENT_DIM,
                "max_seq_len": HISTORY + HORIZON,
                "depth": 1,
                "heads": 2,
                "mlp_dim": 16,
                "dim_head": 4,
            },
        }
    )
    from hydra.utils import instantiate

    model = instantiate(base_cfg)
    assert model.lewm_config is not None

    # 3) Save the diffusion model with the self-contained (inlined) config.
    save_pretrained(
        model,
        run_name="diffusion",
        config=model.export_config(base_cfg),
        filename="weights.pt",
    )

    # 4) Remove the original LeWM checkpoint -- reload must not need it.
    import shutil

    shutil.rmtree(tmp_path / "checkpoints" / "stub_lewm")

    reloaded = load_pretrained("diffusion/weights.pt")

    # State dicts must match exactly (no missing / diverging keys).
    before = model.state_dict()
    after = reloaded.state_dict()
    assert before.keys() == after.keys()
    for key in before:
        torch.testing.assert_close(after[key], before[key])
