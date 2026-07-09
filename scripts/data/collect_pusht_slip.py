"""Collect an intrinsic contact-SLIP PushT dataset — the IRREVERSIBILITY test.

ARC 4 (see experiments/2026-07-01_p0a_mechanism_loop.md). slip16 (ARC 3) showed
the sampling penalty survives extreme REVERSIBLE stochasticity: receding-horizon
replanning re-observes the slip coin before it compounds, so the conditional MEAN
is a sufficient statistic and sampling a generative world model never pays. The
one regime that could break this is IRREVERSIBLE stochasticity — where committing
to the mean cannot be undone by replanning.

PushT provides it: `contact_slip_scale` perturbs the action target by a fair
per-step +-coin; away from the block it is reversible agent wiggle, but AT CONTACT
it changes which side/angle the pusher strikes the T, producing divergent,
path-dependent block poses. p(next | state, action) is then multimodal AND the
modes are irreversible. `contact_slip` is recorded per step for observed-vs-hidden
multimodality screens (mirrors TwoRoom slip_state).

Screen (encoder-free, raw state):
    python scripts/data/multimodality_diagnostic.py --dataset pusht_cslip05.lance \
        --mode dynamics --target-col state --cond-cols state action          # hidden
    python scripts/data/multimodality_diagnostic.py --dataset pusht_cslip05.lance \
        --mode dynamics --target-col state --cond-cols state contact_slip action  # observed
"""

from pathlib import Path

import hydra
import numpy as np
from loguru import logger as logging

import stable_worldmodel as swm
from stable_worldmodel.envs.pusht import WeakPolicy


@hydra.main(version_base=None, config_path='./config', config_name='default')
def run(cfg):
    slip = float(cfg.get('contact_slip_scale', 0.5))
    out_name = cfg.get('out_name', 'pusht_cslip.lance')

    world = swm.World(
        'swm/PushT-v1',
        **cfg.world,
        render_mode='rgb_array',
        contact_slip_scale=slip,
    )
    # WeakPolicy drives the pusher into the block (contact is where the slip
    # branches), giving the aleatoric multimodality the dynamics screen needs.
    world.set_policy(WeakPolicy(dist_constraint=100))

    options = cfg.get('options')
    rng = np.random.default_rng(cfg.seed)

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
        f' 🎉 contact-slip PushT collected (contact_slip_scale={slip}) -> {out}'
    )


if __name__ == '__main__':
    run()
