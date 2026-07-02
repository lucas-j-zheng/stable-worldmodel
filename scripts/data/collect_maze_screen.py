"""Collect a state-only point-maze dataset for the H-JEPA K-step screen.

Random-walk exploration in an OGBench point maze. For the reachability screen
this is the CLEANEST possible data: p(state_{t+K} | state_t) is the K-step
occupancy of a random walker, which branches at wall junctions -- controllable
route multimodality with NO goal confound (no goal-directed policy) and NO
demonstrator-preference confound (uniform route coverage). ob_type='states'
=> no rendering, cheap CPU collection.

Run via SLURM (heavy: rolls thousands of episodes).
"""

import argparse
from pathlib import Path

import stable_worldmodel as swm
from stable_worldmodel.policy import RandomPolicy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=3000)
    ap.add_argument('--maze-type', default='medium',
                    help='arena | medium | large | giant | teleport')
    ap.add_argument('--steps', type=int, default=64, help='max episode steps')
    ap.add_argument('--num-envs', type=int, default=8)
    ap.add_argument('--out', default=None, help='output .lance name')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    out = args.out or f'pointmaze_{args.maze_type}_rand_T{args.steps}.lance'

    world = swm.World(
        'swm/OGBMaze-v0',
        num_envs=args.num_envs,
        add_pixels=False,          # state-only: no rendering, cheap CPU collect
        loco_env_type='point',
        maze_env_type='maze',
        maze_type=args.maze_type,
        ob_type='states',
        max_episode_steps=args.steps,
    )
    world.set_policy(RandomPolicy())

    path = Path(swm.data.utils.get_cache_dir()) / 'datasets' / out
    print(f'[maze-collect] {args.episodes} eps x T={args.steps} '
          f'({args.maze_type}) -> {path}')
    world.collect(path=path, episodes=args.episodes, seed=args.seed)
    print('[maze-collect] done')


if __name__ == '__main__':
    main()
