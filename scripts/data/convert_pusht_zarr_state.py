"""Extract the human-PushT full physical state + action from the DP zarr.

The LeRobot `pusht_image` variant only exposes `proprio` = agent xy (2-D), which
under-conditions the multimodality screen (no block pose). The original Diffusion
Policy zarr (`pusht_cchi_v7_replay.zarr`, Chi et al.) ships the full 5-D state
[agent_x, agent_y, block_x, block_y, block_angle] alongside the 2-D action -- the
clean, encoder-free conditioning we need to compare human vs scripted-expert
policy multimodality on a matched footing.

Writes a small lance (state + action only -- no images) for the existing
diagnostic. Run via SLURM (downloads + decompresses).
"""

import argparse
import os
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import zarr

from stable_worldmodel.data import LanceWriter, get_cache_dir

URL = 'https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='pusht_human_state.lance')
    ap.add_argument('--workdir', default=None, help='staging dir (default: scratch)')
    args = ap.parse_args()

    work = Path(args.workdir or os.environ.get('STABLEWM_HOME', '.')) / 'pusht_dp'
    work.mkdir(parents=True, exist_ok=True)
    zip_path = work / 'pusht.zip'

    zarrs = list(work.rglob('*.zarr'))
    if not zarrs:
        if not zip_path.exists():
            print(f'[zarr] downloading {URL}')
            urlretrieve(URL, zip_path)
        print('[zarr] unzipping')
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(work)
        zarrs = list(work.rglob('*.zarr'))
    zp = zarrs[0]
    print(f'[zarr] opening {zp}')

    root = zarr.open(str(zp), mode='r')
    state = np.asarray(root['data/state'], dtype=np.float32)    # (N, 5)
    action = np.asarray(root['data/action'], dtype=np.float32)  # (N, 2)
    ep_ends = np.asarray(root['meta/episode_ends']).astype(int)  # (E,)
    print(f'[zarr] state={state.shape} action={action.shape} episodes={len(ep_ends)}')

    starts = np.concatenate([[0], ep_ends[:-1]])
    out_path = get_cache_dir(os.environ.get('LOCAL_DATASET_DIR'),
                             sub_folder='datasets') / args.out

    def episode_iter():
        for s, e in zip(starts, ep_ends):
            yield {
                'state': [r for r in state[s:e]],
                'action': [r for r in action[s:e]],
            }

    with LanceWriter(str(out_path), mode='overwrite') as w:
        w.write_episodes(episode_iter())
    print(f'[zarr] wrote {out_path} ({len(ep_ends)} episodes)')


if __name__ == '__main__':
    main()
