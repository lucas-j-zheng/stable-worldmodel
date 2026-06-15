"""Print a dataset's columns and check required keys. Run via SLURM, not login.

    python scripts/data/inspect_dataset.py <path-or-name> [key1 key2 ...]
"""

import sys

import stable_worldmodel as swm


def main():
    if len(sys.argv) < 2:
        raise SystemExit('usage: inspect_dataset.py <path> [required_keys...]')
    name = sys.argv[1]
    required = sys.argv[2:] or [
        'pixels',
        'action',
        'proprio',
        'state',
        'goal_state',
    ]

    ds = swm.data.load_dataset(name, num_steps=4, frameskip=5)
    cols = list(ds.column_names)
    print('INSPECT columns:', cols)
    print('INSPECT len:', len(ds))
    # Presence check only — do NOT get_col_data (would load full columns,
    # e.g. all pixels = many GB). Shapes for small index/state cols come from
    # a single-row read.
    missing = [c for c in required if c not in cols]
    print('INSPECT missing:', missing if missing else 'none')
    try:
        row = ds.get_row_data([0])
        for c in required:
            if c in row:
                v = row[c]
                print(f'INSPECT row0 {c}: shape={getattr(v, "shape", "?")}')
    except Exception as e:  # noqa: BLE001
        print(f'INSPECT row0 read skipped ({type(e).__name__})')


if __name__ == '__main__':
    main()
