"""Fetch a LeWorldModel HF dataset (gated .h5.zst), decompress, and inspect.

The LeWM datasets (quentinll/lewm-*) ship as a single zstd-compressed HDF5
(`*_train.h5.zst`). The repo's HF loader only auto-handles raw `.h5`/`.lance`,
so we download + `zstd -d` here, then print the column schema so downstream
configs (encoder training, latent cache, closed-loop eval) can be wired to the
real layout.

Usage:
    python scripts/data/fetch_lewm_dataset.py \
        --repo quentinll/lewm-pusht --out-dir $STABLEWM_HOME/datasets
"""

import argparse
import subprocess
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True, help='HF dataset repo id')
    ap.add_argument('--out-dir', required=True, help='datasets/ dir to land in')
    ap.add_argument(
        '--keep-zst',
        action='store_true',
        help='keep the compressed file after decompression',
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find the single *.h5.zst (or *.h5) file in the repo.
    files = HfApi().list_repo_files(args.repo, repo_type='dataset')
    cands = [f for f in files if f.endswith(('.h5.zst', '.h5', '.hdf5'))]
    if not cands:
        raise SystemExit(f'No .h5/.h5.zst file in {args.repo}: {files}')
    src = sorted(cands, key=len)[0]
    print(f'[fetch] {args.repo} -> {src}')

    local = hf_hub_download(
        args.repo, src, repo_type='dataset', local_dir=str(out_dir)
    )
    local = Path(local)
    print(f'[fetch] downloaded to {local} ({local.stat().st_size / 1e9:.1f} GB)')

    if local.suffix == '.zst':
        h5 = local.with_suffix('')  # strip .zst -> *.h5
        if h5.exists():
            print(f'[fetch] {h5.name} already decompressed')
        else:
            print(f'[fetch] decompressing -> {h5.name} ...')
            subprocess.run(['zstd', '-d', '-f', str(local), '-o', str(h5)], check=True)
        print(f'[fetch] decompressed {h5} ({h5.stat().st_size / 1e9:.1f} GB)')
        if not args.keep_zst:
            local.unlink()
            print(f'[fetch] removed {local.name}')
    else:
        h5 = local

    # Inspect via the repo's own reader (source of truth for "columns").
    print(f'[inspect] === {h5.name} ===')
    import stable_worldmodel as swm

    ds = swm.data.load_dataset(str(h5), num_steps=4, frameskip=5)
    print('  column_names:', ds.column_names)
    for col in ('state', 'goal_state', 'proprio', 'action', 'pixels'):
        try:
            d = ds.get_col_data(col)
            print(f'  {col}: shape={getattr(d, "shape", "?")} dtype={getattr(d, "dtype", "?")}')
        except Exception as e:  # noqa: BLE001
            print(f'  {col}: MISSING ({type(e).__name__})')
    print('[inspect] done')


if __name__ == '__main__':
    main()
