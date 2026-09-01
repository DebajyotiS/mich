"""Real, preprocessed columnar fMRI data: offline per-run normalization
(`zscore_run`/`preprocess_subject`, driven by `scripts/preprocess_real.py`)
plus the inference-only dataset/datamodule that reads their output.

Each subject's `.npz` (one per subject, written by `preprocess_subject`) holds
a single "bold" array shaped `(L, C, T)` -- every run z-scored independently
across its own time axis, per (layer, column), then concatenated along time.
Unlike `mich.data.synthetic.SyntheticH5Dataset`, there is no ground-truth
"neural"/s/f/v/q/source_position here -- real data has none -- so this only
ever supports the `MICH.forward(bold, time, normalise=True)` inference path,
not the `_shared_step` training/validation path.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

_RUN_RE = re.compile(r"run-(\d+)")


def find_runs(subject_dir: Path) -> list[Path]:
    """Every `sub-*_run-*_bold.npy` file under `subject_dir`, sorted by run number."""
    paths = sorted(
        subject_dir.glob("*_run-*_bold.npy"), key=lambda p: int(_RUN_RE.search(p.name).group(1))
    )
    if not paths:
        raise ValueError(f"No run files found under {subject_dir}")
    return paths


def load_run(path: Path, expected_volumes: int) -> np.ndarray:
    """Load one run's raw `(L, C, T)` array and validate its shape."""
    arr = np.asarray(np.load(path), dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"{path}: expected a 3-D (L, C, T) array, got shape {arr.shape}")
    if arr.shape[-1] != expected_volumes:
        raise ValueError(
            f"{path}: expected {expected_volumes} volumes, got {arr.shape[-1]} "
            f"(shape {arr.shape})"
        )
    return arr


def zscore_run(run: np.ndarray, *, path_for_error: Path) -> np.ndarray:
    """Z-score `run` (L, C, T) across its own time axis, per (layer, column).

    Raises:
        ValueError: If any (layer, column) has ~zero std over this run --
            fail loudly rather than silently dividing by a near-zero number
            or masking the column out.
    """
    mean = run.mean(axis=-1, keepdims=True)
    std = run.std(axis=-1, keepdims=True)
    bad = np.argwhere(std[..., 0] < 1e-8)
    if bad.size > 0:
        layer, col = bad[0]
        raise ValueError(
            f"{path_for_error}: near-zero std at layer={layer}, column={col} "
            f"(std={std[layer, col, 0]:.3g}) -- refusing to z-score a constant "
            "column silently."
        )
    return (run - mean) / std


def preprocess_subject(
    subject_dir: Path, *, expected_volumes: int
) -> tuple[np.ndarray, list[int]]:
    """Load, per-run z-score, and concatenate every run for one subject.

    Runs are z-scored *before* concatenation (not concatenated then z-scored
    as a whole) so that between-run scanner/drift differences are removed
    before runs are pooled.

    Returns:
        (bold, run_lengths): bold is `(L, C, T_total)` float32, run_lengths
        is the number of volumes each run contributed, in concatenation order.
    """
    run_paths = find_runs(subject_dir)
    normalized_runs = []
    run_lengths = []
    for run_path in run_paths:
        run = load_run(run_path, expected_volumes)
        normalized_runs.append(zscore_run(run, path_for_error=run_path))
        run_lengths.append(run.shape[-1])
    bold = np.concatenate(normalized_runs, axis=-1).astype(np.float32)
    return bold, run_lengths


class RealColumnarDataset(Dataset):
    """One sample per subject `.npz` file (see module docstring for the format
    `preprocess_subject` writes: a single "bold" array shaped `(L, C, T)`,
    already per-run z-scored and concatenated across runs).

    Returns:
        {"bold": [L, T, C, 1] float32 tensor, "subject_id": str} -- matching
        `MICH.forward`'s `[B, L, T, H, W]` contract with H=C (real columns)
        and a trailing degenerate W=1 axis (see config/simulation/columnar.yaml),
        so `bold` can be fed straight into `MICH.forward` unchanged.
    """

    def __init__(self, paths: Sequence[str]):
        super().__init__()
        self.paths = [Path(p) for p in paths]
        if not self.paths:
            raise ValueError("RealColumnarDataset received no files")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.paths[idx]
        with np.load(path) as npz:
            bold = np.asarray(npz["bold"], dtype=np.float32)  # (L, C, T)
        if bold.ndim != 3:
            raise ValueError(f"{path}: expected bold.ndim == 3 (L, C, T), got shape {bold.shape}")
        bold = np.ascontiguousarray(bold.transpose(0, 2, 1))[..., None]  # (L, T, C, 1)
        return {
            "bold": torch.from_numpy(bold),
            "subject_id": path.stem,
        }


class RealColumnarDataModule(pl.LightningDataModule):
    """Inference-only datamodule over a directory of `scripts/preprocess_real.py`
    output files.

    There is no train/val/test split -- real data carries no ground truth to
    hold anything out for -- so `test_dataloader`/`predict_dataloader` both
    serve every subject found under `data.path`.

    Batching across subjects requires every subject to have the same number
    of columns `C` (the default DataLoader collate function stacks tensors
    and will raise otherwise); `loader.batch_size` defaults to 1 to sidestep
    this until that's confirmed true of the real data.
    """

    def __init__(self, data: Mapping[str, Any], loader: Mapping[str, Any]):
        super().__init__()
        self.data_config = dict(data)
        self.loader_config = dict(loader)
        self.ds: RealColumnarDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        data_dir = self.data_config.get("path")
        if data_dir is None:
            raise ValueError("data.path must be set")
        paths = sorted(Path(data_dir).glob("*.npz"))
        if not paths:
            raise ValueError(f"No .npz files found in {data_dir}")
        self.ds = RealColumnarDataset([str(p) for p in paths])

    def _make_loader(self) -> DataLoader:
        assert self.ds is not None, "setup() must be called before requesting a dataloader"
        return DataLoader(
            self.ds,
            batch_size=int(self.loader_config.get("batch_size", 1)),
            shuffle=False,
            drop_last=False,
            num_workers=int(self.loader_config.get("num_workers", 0)),
            pin_memory=bool(self.loader_config.get("pin_memory", False)),
        )

    def test_dataloader(self) -> DataLoader:
        return self._make_loader()

    def predict_dataloader(self) -> DataLoader:
        return self._make_loader()
