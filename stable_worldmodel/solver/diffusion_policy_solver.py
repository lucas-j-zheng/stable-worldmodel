"""Solver wrapper that turns a trained DiffusionPolicy/MLPPolicy into actions.

Drop-in for CEMSolver in the eval harness: it implements `configure` + `solve`
so WorldModelPolicy's history/goal/transform machinery is reused verbatim. There
is NO planning/CEM here -- the policy directly samples an action chunk from
(latent history, goal latent), which is the whole point (a generative policy vs a
planner over a dynamics model).
"""

import torch


class DiffusionPolicySolver:
    def __init__(self, model, device: str | torch.device = 'cuda', **kwargs):
        self.model = model.eval()
        self.device = device
        try:
            self.model.to(device)
        except Exception:  # pragma: no cover
            pass

    def configure(self, *, action_space, n_envs, config):
        self._action_space = action_space
        self._n_envs = n_envs
        self._config = config
        self._configured = True

    @property
    def history_size(self) -> int:
        return self.model.history_size

    @torch.inference_mode()
    def _encode(self, pixels):
        emb = self.model.encode_latents({'pixels': pixels.to(self.device)})
        return emb

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action=None) -> dict:
        # pixels: (n, history_len, C, H, W); goal: (n, T, C, H, W).
        pixels = info_dict['pixels']
        goal = info_dict['goal']
        if not torch.is_tensor(pixels):
            pixels = torch.as_tensor(pixels)
        if not torch.is_tensor(goal):
            goal = torch.as_tensor(goal)

        hist = self._encode(pixels)                       # (n, hist_len, latent)
        hist = hist[:, -self.history_size:]
        goal_emb = self._encode(goal)[:, -1]              # (n, latent)

        actions = self.model.sample_actions(hist, goal_emb)   # (n, horizon, act_blk)
        return {'actions': actions.detach().cpu().float()}

    def __call__(self, *args, **kwargs):
        return self.solve(*args, **kwargs)
