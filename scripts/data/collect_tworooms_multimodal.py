"""Collect a DELIBERATELY MULTIMODAL TwoRoom dataset.

Two symmetric, equally-good doors in the wall + a stochastic expert that commits
to a RANDOM door per episode (independent of the observable state). So the same
(state) maps to different action sequences across episodes = designed, hard
multimodality -- the regime where a diffusion world-model/policy should beat a
deterministic predictor that is forced to mode-average. Use this to validate the
whole diffusion-vs-deterministic apparatus where multimodality provably exists
(the structural screen will confirm residual_bimodal jumps vs the greedy expert).
"""

from pathlib import Path

import hydra
import numpy as np
from loguru import logger as logging

import stable_worldmodel as swm
from stable_worldmodel.envs.two_room import ExpertPolicy


@hydra.main(version_base=None, config_path='./config', config_name='default')
def run(cfg):
    world = swm.World('swm/TwoRoom-v1', **cfg.world, render_mode='rgb_array')
    world.set_policy(
        ExpertPolicy(
            action_noise=2.0,
            action_repeat_prob=0.05,
            stochastic_door=True,   # random committed door per episode
            seed=cfg.seed,
        )
    )

    rng = np.random.default_rng(cfg.seed)
    # Fixed 2-door symmetric geometry (about the vertical wall ~y=112); random
    # agent/target each episode. Both doors fit (size 14 half-extent >> 1.1*radius
    # with radius=7), so the expert genuinely chooses between them.
    options = {
        'variation': ['agent.position', 'target.position'],
        'variation_values': {
            'door.number': 2,
            'door.position': [60, 164, 49],
            'door.size': [14, 14, 14],
        },
    }

    out = (
        Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
        / 'datasets'
        / 'tworoom_multimodal.lance'
    )
    world.collect(
        out,
        episodes=cfg.num_traj,
        seed=rng.integers(0, 1_000_000).item(),
        options=options,
    )
    logging.success(f' 🎉 multimodal tworoom collected -> {out}')


if __name__ == '__main__':
    run()
