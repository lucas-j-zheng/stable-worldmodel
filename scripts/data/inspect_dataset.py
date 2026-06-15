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
    print('INSPECT columns:', ds.column_names)
    print('INSPECT len:', len(ds))
    missing = []
    for c in required:
        try:
            d = ds.get_col_data(c)
            print(f'INSPECT col {c}: shape={getattr(d, "shape", "?")} '
                  f'dtype={getattr(d, "dtype", "?")}')
        except Exception as e:  # noqa: BLE001
            print(f'INSPECT col {c}: MISSING ({type(e).__name__})')
            missing.append(c)
    print('INSPECT missing:', missing if missing else 'none')


if __name__ == '__main__':
    main()
