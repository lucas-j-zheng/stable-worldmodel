"""Collect a PARTIALLY-OBSERVED (POMDP) TwoRoom dataset with MULTIMODAL dynamics.

The point of this dataset is to make the *dynamics* p(next | obs, action)
multimodal -- the half of the thesis the policy experiments never tested. The
mechanism is partial observability, not stochastic physics:

  - TwoRoom motion is deterministic EXCEPT at the wall, where pass-through vs.
    bounce depends on the DOOR configuration.
  - The observation the model sees is `state` = agent position only; the door
    layout is part of the world but NOT observed.
  - Here we RANDOMIZE the (hidden) door layout per episode (number + positions),
    so at a fixed observation (agent near the wall, action into the wall) the
    next state is genuinely bimodal across episodes: pass if a hidden door is
    there, bounce if not. Condition on `state`+`door_state` (full obs) and it
    collapses to deterministic. That contrast is the experiment.

We drive with a HIGH-action-noise expert (near-random walk) so the agent hits
the wall at random points and angles, sampling BOTH outcomes (pass and bounce)
at matched (agent_pos, action) -- a goal-directed greedy expert would aim only
at its episode's door and undersample the bounce mode.

The hidden door layout is recorded as the `door_state` column (see
TwoRoomEnv._get_info) so the screen can run the with/without-doors contrast
encoder-free:
    python scripts/data/multimodality_diagnostic.py --dataset tworoom_pomdp.lance \
        --mode dynamics --target-col state --cond-cols state action          # hidden
    python scripts/data/multimodality_diagnostic.py --dataset tworoom_pomdp.lance \
        --mode dynamics --target-col state --cond-cols state door_state action # observed
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

    # High action noise -> near-random exploration of the wall region so the
    # dataset contains pass AND bounce at matched (agent_pos, action). The unit
    # direction is mostly swamped, which is exactly what we want for dynamics
    # coverage (opposite of the policy-multimodality collection).
    action_noise = float(cfg.get('action_noise', 2.0))
    out_name = cfg.get('out_name', 'tworoom_pomdp.lance')
    world.set_policy(
        ExpertPolicy(
            action_noise=action_noise,
            action_repeat_prob=0.0,
            seed=cfg.seed,
        )
    )

    rng = np.random.default_rng(cfg.seed)
    # Randomize the HIDDEN door layout per episode: number (1-3) and positions.
    # Keep size fixed (half-extent 14 >> 1.1*radius=7.7) so >=1 door always fits
    # and the env stays solvable. agent/target also randomized each episode.
    options = {
        'variation': [
            'agent.position',
            'target.position',
            'door.number',
            'door.position',
        ],
        'variation_values': {
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
        f' 🎉 POMDP TwoRoom collected (action_noise={action_noise}, '
        f'random hidden doors) -> {out}'
    )


if __name__ == '__main__':
    run()
