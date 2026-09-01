"""CLI wrapper for offline preprocessing of real columnar fMRI data -- mirrors
run_sim.py being a separate offline step from train_mich.py.

Expected raw input layout (one file per subject-run; convention TBD with the
column-extraction pipeline's owner -- adjust `mich.data.real.find_runs`/
`load_run` once the real convention is confirmed):

    <input_dir>/sub-<NN>/sub-<NN>_run-<K>_bold.npy

each holding a single `(L, C, T)` array (L=3 layers, C columns, T volumes).

Every run is z-scored independently across time (per layer, per column, over
that run's own T volumes) and then concatenated along time -- see
`mich.data.real.preprocess_subject` for the actual logic and
`mich.data.real.zscore_run` for why this is per-run-then-concatenate, not
concatenate-then-z-score. Output is one `<output_dir>/sub-<NN>.npz` per
subject holding the concatenated, already-normalized `(L, C, T_total)` "bold"
array plus "run_lengths" (volumes contributed by each run, in concatenation
order).

Fails loudly (does not mask/skip) on a run with an unexpected volume count or
a column with ~zero variance -- this is 21-subject clinical-scale data, and
silently dropping or flattening a bad column risks hiding real data-quality
problems rather than surfacing them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mich.data.real import preprocess_subject


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, required=True, help="Directory of sub-<NN>/ raw run files"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Where to write sub-<NN>.npz files"
    )
    parser.add_argument(
        "--expected-volumes", type=int, default=162, help="Required T per run (default 162)"
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    subject_dirs = sorted(p for p in args.input_dir.glob("sub-*") if p.is_dir())
    if not subject_dirs:
        raise ValueError(f"No sub-*/ directories found under {args.input_dir}")

    for subject_dir in subject_dirs:
        bold, run_lengths = preprocess_subject(subject_dir, expected_volumes=args.expected_volumes)
        out_path = args.output_dir / f"{subject_dir.name}.npz"
        np.savez(out_path, bold=bold, run_lengths=np.asarray(run_lengths, dtype=np.int64))
        print(f"{subject_dir.name}: {len(run_lengths)} runs -> bold {bold.shape} -> {out_path}")


if __name__ == "__main__":
    main()
