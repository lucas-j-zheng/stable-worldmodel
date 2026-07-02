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
    # action_noise=0: TwoRoom actions are unit-magnitude direction vectors, so
    # std-2.0 Gaussian noise (the default collect setting) is mostly noise and
    # SWAMPS the door-choice modes (separation ~1) into overlap -> the screen
    # then reads unimodal. Keep it clean so the two routes are separable modes.
    # stochastic_door / out_name overridable so the same 2-door geometry yields
    # both the multimodal (random door) and greedy (closest door) baselines.
    stochastic = bool(cfg.get('stochastic_door', True))
    out_name = cfg.get('out_name', 'tworoom_multimodal.lance')
    door_prob = float(cfg.get('door_prob', 0.5))   # dose knob (0.5 bimodal..1 unimodal)
    world.set_policy(
        ExpertPolicy(
            action_noise=0.0,
            action_repeat_prob=0.0,
            stochastic_door=stochastic,
            door_prob=door_prob,
            seed=cfg.seed,
        )
    )

    rng = np.random.default_rng(cfg.seed)
    # Fixed symmetric door geometry (about the vertical wall ~y=112); random
    # agent/target each episode. All doors fit (size 14 half-extent >> 1.1*radius
    # with radius=7), so the expert genuinely chooses between them.
    # n_doors=3 adds a CENTER door at 112: the trimodal falsification test of
    # the mean-in-gap mechanism (H-JEPA review E14) -- with a center route the
    # conditional mean lies ON a real mode, so MSE should NOT be punished
    # despite >=2-door multimodality. (Expert commits uniformly among >2 doors.)
    n_doors = int(cfg.get('n_doors', 2))
    positions = {2: [60, 164, 49], 3: [40, 112, 184]}[n_doors]
    options = {
        'variation': ['agent.position', 'target.position'],
        'variation_values': {
            'door.number': n_doors,
            'door.position': positions,
            'door.size': [14, 14, 14],
        },
    }

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
        f' 🎉 tworoom collected (stochastic_door={stochastic}) -> {out}'
    )


if __name__ == '__main__':
    run()
