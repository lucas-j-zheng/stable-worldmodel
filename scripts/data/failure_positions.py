"""P0a failure-position probe: where do FAILED episodes end?

Reads the positions_off{K}_seed{S}.npz files written by eval_wm.py (per-step
agent positions + per-episode success + goal positions) for a set of policy
checkpoints, and classifies each FAILED episode's final position:

  - wall_between: within WALL_BAND px of the wall line (112) AND the along-wall
    coordinate strictly between the two door spans (74..150) -> the
    mode-averaging signature (driving at the wall between the doors).
  - wall_door: within the band at a door span (stuck in/near a doorway).
  - elsewhere: everything else (wrong room, ran out of budget short of goal...).

Geometry (fixed by the mm collection + eval options): wall line at 112, doors
centered 60/164 half-extent 14. The env's wall axis is index-ambiguous here, so
both axis conventions are computed; the one with mass is the real one (videos
disambiguate). Also reports mean final distance-to-goal on failures.

Usage:
  python scripts/data/failure_positions.py --ckpt-root /path/to/checkpoints \
      --models dp_mm05_ps1 dp_mm05_ps2 ... --group-len 2
Groups are inferred from model prefix before the last '_ps' (dp/tmse pooled
across seeds and eval-seed files).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

WALL_POS = 112.0
WALL_BAND = 25.0
DOOR_SPANS = [(46.0, 74.0), (150.0, 178.0)]
BETWEEN = (74.0, 150.0)


def classify(final_pos, axis):
    wall_c = final_pos[axis]
    along_c = final_pos[1 - axis]
    if abs(wall_c - WALL_POS) < WALL_BAND:
        if BETWEEN[0] < along_c < BETWEEN[1]:
            return 'wall_between'
        if any(lo <= along_c <= hi for lo, hi in DOOR_SPANS):
            return 'wall_door'
        return 'wall_outer'
    return 'elsewhere'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-root', required=True)
    ap.add_argument('--models', nargs='+', required=True)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    root = Path(args.ckpt_root)
    stats = defaultdict(lambda: defaultdict(int))
    dists = defaultdict(list)

    for model in args.models:
        group = model.split('_ps')[0].split('_s')[0]
        for npz_path in sorted(root.glob(f'{model}/positions_*.npz')):
            d = np.load(npz_path)
            traj = d['trajectories']          # (n, T, 2)
            succ = d['successes'].astype(bool)
            goals = d['goal_proprio']          # (n, 2) or empty
            finals = traj[:, -1]
            for i in range(len(succ)):
                key = 'success' if succ[i] else 'fail'
                stats[group][f'n_{key}'] += 1
                if succ[i]:
                    continue
                for axis in (0, 1):
                    c = classify(finals[i], axis)
                    stats[group][f'ax{axis}_{c}'] += 1
                if goals.size:
                    dists[group].append(
                        float(np.linalg.norm(finals[i] - goals[i])))

    out = {}
    for group, s in sorted(stats.items()):
        nf = s['n_fail'] or 1
        row = {k: v for k, v in sorted(s.items())}
        row['fail_frac_wall_between_ax0'] = s['ax0_wall_between'] / nf
        row['fail_frac_wall_between_ax1'] = s['ax1_wall_between'] / nf
        row['fail_mean_dist_to_goal'] = (
            float(np.mean(dists[group])) if dists[group] else None)
        out[group] = row
        print(f"[failpos] {group}: fails={s['n_fail']}/"
              f"{s['n_fail'] + s['n_success']}  "
              f"wall_between ax0={row['fail_frac_wall_between_ax0']:.2f} "
              f"ax1={row['fail_frac_wall_between_ax1']:.2f}  "
              f"door ax0={s['ax0_wall_door'] / nf:.2f} "
              f"ax1={s['ax1_wall_door'] / nf:.2f}  "
              f"mean_dist_goal={row['fail_mean_dist_to_goal']}")

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'[failpos] wrote {args.out}')


if __name__ == '__main__':
    main()
