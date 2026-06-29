"""Collect a hidden-DRIFT POMDP TwoRoom dataset with PERVASIVE multimodal dynamics.

Operationalization #2 of the POMDP-dynamics experiment (see
experiments/2026-06-28_pomdp_dynamics.md). The hidden-doors version (#1) failed
because the door ambiguity bites only in a thin sliver at the wall -> aggregate
dynamics stay deterministic (det_R2 0.966). Here the hidden variable bites
EVERYWHERE:

  Each episode samples a constant, UNOBSERVED drift ("wind") in {-d, +d} along x;
  the agent steps as next = pos + action*speed + drift. With drift hidden and
  resampled per episode, the same (observation, action) splits into two separated
  next-state branches on EVERY step -> pervasive bimodal dynamics. A deterministic
  predictor must average the branches; a diffusion model can commit to one. This
  is the fair test of the DYNAMICS half of the thesis.

drift_scale is passed as an env kwarg (forwarded by World -> gym.make -> the env);
the per-episode sign is drawn in TwoRoomEnv.reset and recorded as `drift_state`.
The geometry is the simple default (1 door, random agent/target); a high-noise
exploratory policy gives broad free-space coverage where the drift acts.

Screen (encoder-free):
    python scripts/data/multimodality_diagnostic.py --dataset tworoom_drift.lance \
        --mode dynamics --target-col state --cond-cols state action          # hidden
    python scripts/data/multimodality_diagnostic.py --dataset tworoom_drift.lance \
        --mode dynamics --target-col state --cond-cols state drift_state action # observed
"""

from pathlib import Path

import hydra
import numpy as np
from loguru import logger as logging

import stable_worldmodel as swm
from stable_worldmodel.envs.two_room import ExpertPolicy


@hydra.main(version_base=None, config_path='./config', config_name='default')
def run(cfg):
    drift_scale = float(cfg.get('drift_scale', 3.0))
    # drift_scale flows World(**) -> gym.make(**kwargs) -> TwoRoomEnv.__init__.
    world = swm.World(
        'swm/TwoRoom-v1',
        **cfg.world,
        render_mode='rgb_array',
        drift_scale=drift_scale,
    )

    action_noise = float(cfg.get('action_noise', 2.0))
    out_name = cfg.get('out_name', 'tworoom_drift.lance')
    world.set_policy(
        ExpertPolicy(
            action_noise=action_noise,
            action_repeat_prob=0.0,
            seed=cfg.seed,
        )
    )

    rng = np.random.default_rng(cfg.seed)
    # Simple default geometry; drift (not doors) is the hidden multimodality.
    options = {'variation': ['agent.position', 'target.position']}

    out = (
        Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
        / 'datasets'
        / out_name
    )
    world.collect(
        out,
        episodes=cfg.num_traj,
        seed=rng.integers(0, 1_000_000).item(),
        options=options,
    )
    logging.success(
        f' 🎉 drift POMDP TwoRoom collected (drift_scale={drift_scale}, '
        f'action_noise={action_noise}) -> {out}'
    )


if __name__ == '__main__':
    run()
