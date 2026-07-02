"""Collect an intrinsic-SLIP TwoRoom dataset with genuinely multimodal dynamics.

P1 Route 1 (see todos/diffusion.md): the POMDP routes failed because partial
observability in a bounded deterministic env could not create substantial
multimodal dynamics (hidden doors too sparse; hidden drift resolvable from
history AND clamp-absorbed at magnitude). Here the stochasticity is INTRINSIC
and per-step:

  Every step draws a fair coin and displaces the agent by +-slip_scale along
  the along-wall axis BEFORE collision handling. p(next | state, action) is a
  two-point mixture separated by 2*slip_scale on EVERY step, and no
  conditioning (history or otherwise) can resolve the coin. This is the
  aleatoric branching the thesis' dynamics half needs; slip_scale is the dose.

slip_scale flows World(**) -> gym.make(**kwargs) -> TwoRoomEnv.__init__;
the realized slip is recorded per step as `slip_state`.

Screen (encoder-free):
    python scripts/data/multimodality_diagnostic.py --dataset tworoom_slip4.lance \
        --mode dynamics --target-col state --cond-cols state action        # hidden
    python scripts/data/multimodality_diagnostic.py --dataset tworoom_slip4.lance \
        --mode dynamics --target-col state --cond-cols state slip_state action # observed
"""

from pathlib import Path

import hydra
import numpy as np
from loguru import logger as logging

import stable_worldmodel as swm
from stable_worldmodel.envs.two_room import ExpertPolicy


@hydra.main(version_base=None, config_path='./config', config_name='default')
def run(cfg):
    slip_scale = float(cfg.get('slip_scale', 4.0))
    world = swm.World(
        'swm/TwoRoom-v1',
        **cfg.world,
        render_mode='rgb_array',
        slip_scale=slip_scale,
    )

    action_noise = float(cfg.get('action_noise', 2.0))
    out_name = cfg.get('out_name', 'tworoom_slip.lance')
    world.set_policy(
        ExpertPolicy(
            action_noise=action_noise,
            action_repeat_prob=0.0,
            seed=cfg.seed,
        )
    )

    rng = np.random.default_rng(cfg.seed)
    # Simple default geometry; the slip (not doors) is the multimodality source.
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
        f' 🎉 slip TwoRoom collected (slip_scale={slip_scale}, '
        f'action_noise={action_noise}) -> {out}'
    )


if __name__ == '__main__':
    run()
