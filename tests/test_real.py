"""Tests for src/mich/data/real.py -- real-data preprocessing and the
inference-only dataset/datamodule."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mich.data.real import (
    RealColumnarDataModule,
    RealColumnarDataset,
    find_runs,
    load_run,
    preprocess_subject,
    zscore_run,
)

# -------------------------
# zscore_run / preprocess_subject
# -------------------------


def test_zscore_run_produces_zero_mean_unit_std_per_layer_column():
    rng = np.random.default_rng(0)
    run = rng.normal(loc=5.0, scale=3.0, size=(2, 4, 50))
    z = zscore_run(run, path_for_error="dummy")
    assert np.allclose(z.mean(axis=-1), 0.0, atol=1e-10)
    assert np.allclose(z.std(axis=-1), 1.0, atol=1e-10)


def test_zscore_run_raises_on_constant_column():
    run = np.ones((1, 2, 10))
    run[0, 1, :] = np.arange(10)  # only column 0 is constant
    with pytest.raises(ValueError, match="near-zero std"):
        zscore_run(run, path_for_error="dummy")


def test_load_run_raises_on_wrong_volume_count(tmp_path):
    path = tmp_path / "sub-01_run-1_bold.npy"
    np.save(path, np.zeros((3, 5, 100)))
    with pytest.raises(ValueError, match="expected 162 volumes"):
        load_run(path, expected_volumes=162)


def test_find_runs_sorts_by_run_number(tmp_path):
    subject_dir = tmp_path / "sub-01"
    subject_dir.mkdir()
    for k in [3, 1, 2]:
        np.save(subject_dir / f"sub-01_run-{k}_bold.npy", np.zeros((3, 2, 5)))
    runs = find_runs(subject_dir)
    assert [p.name for p in runs] == [
        "sub-01_run-1_bold.npy",
        "sub-01_run-2_bold.npy",
        "sub-01_run-3_bold.npy",
    ]


def test_preprocess_subject_zscores_before_concatenating_not_after(tmp_path):
    """Two runs with deliberately different mean/scale: z-score-per-run-then-
    concatenate must fully remove each run's own offset/scale (every run ends
    up zero-mean/unit-std on its own), which concatenate-then-z-score would
    NOT do (the pooled z-score would leave a visible step between runs, since
    a single global mean/std can't zero both runs' means simultaneously)."""
    subject_dir = tmp_path / "sub-01"
    subject_dir.mkdir()
    rng = np.random.default_rng(1)
    run1 = rng.normal(loc=100.0, scale=1.0, size=(2, 3, 20))  # high mean, small scale
    run2 = rng.normal(loc=-50.0, scale=10.0, size=(2, 3, 20))  # low mean, large scale
    np.save(subject_dir / "sub-01_run-1_bold.npy", run1)
    np.save(subject_dir / "sub-01_run-2_bold.npy", run2)

    bold, run_lengths = preprocess_subject(subject_dir, expected_volumes=20)

    assert bold.shape == (2, 3, 40)
    assert run_lengths == [20, 20]
    first_run_out = bold[..., :20]
    second_run_out = bold[..., 20:]
    # each run independently zero-mean/unit-std -> no visible step at the seam
    assert np.allclose(first_run_out.mean(axis=-1), 0.0, atol=1e-5)
    assert np.allclose(second_run_out.mean(axis=-1), 0.0, atol=1e-5)
    # concatenate-then-z-score would NOT satisfy this: the pooled std would be
    # dominated by the ~150-unit gap between run means, so run1 alone would
    # collapse to a much smaller-than-1 apparent std under a single global scale.
    assert np.allclose(first_run_out.std(axis=-1), 1.0, atol=1e-1)
    assert np.allclose(second_run_out.std(axis=-1), 1.0, atol=1e-1)


def test_preprocess_subject_raises_on_run_with_wrong_volume_count(tmp_path):
    subject_dir = tmp_path / "sub-01"
    subject_dir.mkdir()
    np.save(subject_dir / "sub-01_run-1_bold.npy", np.random.randn(2, 3, 10))
    with pytest.raises(ValueError, match="expected 5 volumes"):
        preprocess_subject(subject_dir, expected_volumes=5)


# -------------------------
# RealColumnarDataset
# -------------------------


def test_dataset_returns_degenerate_width_axis_and_subject_id(tmp_path):
    path = tmp_path / "sub-01.npz"
    bold = np.random.randn(3, 7, 40).astype(np.float32)
    np.savez(path, bold=bold, run_lengths=np.array([40]))

    ds = RealColumnarDataset([str(path)])
    assert len(ds) == 1
    item = ds[0]
    assert item["subject_id"] == "sub-01"
    assert isinstance(item["bold"], torch.Tensor)
    assert item["bold"].shape == (3, 40, 7, 1)
    assert torch.allclose(item["bold"][:, :, :, 0], torch.from_numpy(bold).transpose(1, 2))


def test_dataset_raises_on_empty_paths():
    with pytest.raises(ValueError, match="no files"):
        RealColumnarDataset([])


def test_dataset_raises_on_non_3d_bold(tmp_path):
    path = tmp_path / "sub-01.npz"
    np.savez(path, bold=np.zeros((3, 7)))
    ds = RealColumnarDataset([str(path)])
    with pytest.raises(ValueError, match="ndim == 3"):
        ds[0]


# -------------------------
# RealColumnarDataModule
# -------------------------


def _write_subject_npz(path, *, L=3, C=5, T=20):
    np.savez(path, bold=np.random.randn(L, C, T).astype(np.float32), run_lengths=np.array([T]))


def test_datamodule_discovers_all_npz_files_and_serves_them(tmp_path):
    for name in ["sub-01", "sub-02", "sub-03"]:
        _write_subject_npz(tmp_path / f"{name}.npz")

    dm = RealColumnarDataModule(data={"path": str(tmp_path)}, loader={"batch_size": 1})
    dm.setup()
    assert len(dm.ds) == 3

    loader = dm.test_dataloader()
    batches = list(loader)
    assert len(batches) == 3
    assert dm.predict_dataloader() is not None


def test_datamodule_raises_when_path_missing():
    dm = RealColumnarDataModule(data={}, loader={})
    with pytest.raises(ValueError, match="data.path must be set"):
        dm.setup()


def test_datamodule_raises_when_no_npz_files(tmp_path):
    dm = RealColumnarDataModule(data={"path": str(tmp_path)}, loader={})
    with pytest.raises(ValueError, match="No .npz files found"):
        dm.setup()
