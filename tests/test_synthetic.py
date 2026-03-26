"""Tests for src/data/synthetic.py — dtype helpers, split logic, dataset, and DataModule."""

from __future__ import annotations

import pickle

import h5py
import numpy as np
import pytest
import torch

from src.data.synthetic import (
    SyntheticDataModule,
    SyntheticH5Dataset,
    _compute_split_counts,
    _np_dtype,
    _torch_dtype,
)

# -------------------------
# Helpers
# -------------------------

_CACHE_CFG: dict = {}


def _make_h5(
    path,
    *,
    n: int = 5,
    t: int = 8,
    h: int = 4,
    w: int = 4,
    layers: tuple = ("layer_0", "layer_1"),
    include_meta: bool = True,
    include_latents: bool = True,
    latent_t: int | None = None,
) -> None:
    """Write a minimal valid HDF5 file understood by SyntheticH5Dataset."""
    lt = latent_t if latent_t is not None else t
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        for lyr in layers:
            grp = f.require_group(lyr)
            grp.create_dataset("bold", data=rng.standard_normal((n, t, h, w)).astype(np.float32))
            grp.create_dataset("x", data=rng.standard_normal((n, t, h, w)).astype(np.float32))
            for key in ("s", "f", "v", "q", "v_star", "q_star"):
                grp.create_dataset(key, data=rng.standard_normal((n, lt, h, w)).astype(np.float32))
        if include_meta:
            meta = f.require_group("meta")
            meta.create_dataset("num_pulses", data=rng.integers(1, 4, size=n).astype(np.int32))
            meta.create_dataset(
                "source_layer", data=rng.integers(0, len(layers), size=n).astype(np.int32)
            )
            meta.create_dataset(
                "source_position",
                data=rng.integers(0, min(h, w), size=(n, 2)).astype(np.int32),
            )


# -------------------------
# _torch_dtype
# -------------------------


@pytest.mark.parametrize(
    "s, expected",
    [
        ("float16", torch.float16),
        ("fp16", torch.float16),
        ("half", torch.float16),
        ("float32", torch.float32),
        ("fp32", torch.float32),
        ("float", torch.float32),
        ("FLOAT32", torch.float32),  # case-insensitive
        ("FP16", torch.float16),
    ],
)
def test_torch_dtype_known_aliases(s, expected):
    assert _torch_dtype(s) == expected


def test_torch_dtype_unknown_raises():
    with pytest.raises(ValueError, match="Unsupported dtype"):
        _torch_dtype("bfloat16")


# -------------------------
# _np_dtype
# -------------------------


def test_np_dtype_float16():
    assert _np_dtype(torch.float16) == np.float16


def test_np_dtype_float32():
    assert _np_dtype(torch.float32) == np.float32


def test_np_dtype_unsupported_raises():
    with pytest.raises(ValueError, match="Unsupported torch dtype"):
        _np_dtype(torch.float64)


# -------------------------
# _compute_split_counts — fraction-based
# -------------------------


def test_split_fracs_default_sums_to_n():
    tr, va, te = _compute_split_counts(100, {})
    assert tr + va + te == 100


def test_split_fracs_explicit_splits_correct():
    tr, va, te = _compute_split_counts(100, {"val_frac": 0.1, "test_frac": 0.2})
    assert va == 10
    assert te == 20
    assert tr == 70
    assert tr + va + te == 100


def test_split_fracs_sum_gt_one_raises():
    with pytest.raises(ValueError, match="val_frac.*test_frac|test_frac.*val_frac"):
        _compute_split_counts(100, {"val_frac": 0.6, "test_frac": 0.6})


def test_split_fracs_negative_raises():
    with pytest.raises(ValueError, match="non-negative"):
        _compute_split_counts(100, {"val_frac": -0.1, "test_frac": 0.1})


def test_split_train_frac_explicit_sums_to_n():
    tr, va, te = _compute_split_counts(100, {"train_frac": 0.8, "val_frac": 0.1, "test_frac": 0.1})
    assert tr + va + te == 100
    assert tr == 80
    assert va == 10


def test_split_train_frac_not_sum_to_one_raises():
    with pytest.raises(ValueError, match="sum to 1"):
        _compute_split_counts(100, {"train_frac": 0.7, "val_frac": 0.1, "test_frac": 0.1})


@pytest.mark.parametrize("n", [1, 2, 3, 10])
def test_split_fracs_small_n_sums_to_n(n):
    tr, va, te = _compute_split_counts(n, {"val_frac": 0.2, "test_frac": 0.2})
    assert tr + va + te == n


# -------------------------
# _compute_split_counts — count-based
# -------------------------


def test_split_counts_explicit_values():
    tr, va, te = _compute_split_counts(100, {"train_count": 70, "val_count": 20, "test_count": 10})
    assert tr == 70
    assert va == 20
    assert te == 10


def test_split_counts_exceed_n_raises():
    with pytest.raises(ValueError, match="exceed dataset size"):
        _compute_split_counts(100, {"train_count": 80, "val_count": 15, "test_count": 10})


def test_split_counts_missing_train_allocates_remainder():
    tr, va, te = _compute_split_counts(100, {"val_count": 20, "test_count": 10})
    assert va == 20
    assert te == 10
    assert tr == 70


def test_split_counts_take_precedence_over_fracs():
    # Fraction would give val=50, but explicit count overrides
    tr, va, te = _compute_split_counts(
        100,
        {"train_count": 60, "val_count": 10, "test_count": 5, "val_frac": 0.5},
    )
    assert va == 10


# -------------------------
# SyntheticH5Dataset — basic properties
# -------------------------


def test_dataset_len_returns_n(tmp_path):
    p = str(tmp_path / "d.h5")
    _make_h5(p, n=7)
    ds = SyntheticH5Dataset(p, cache_cfg=_CACHE_CFG, layers=("layer_0", "layer_1"))
    assert len(ds) == 7


def test_dataset_getitem_returns_bold_and_neural(tmp_path):
    p = str(tmp_path / "d.h5")
    _make_h5(p, n=3, t=8, h=4, w=4, layers=("layer_0", "layer_1"))
    ds = SyntheticH5Dataset(p, cache_cfg=_CACHE_CFG, layers=("layer_0", "layer_1"))
    item = ds[0]
    assert "bold" in item and "neural" in item
    assert item["bold"].shape == (2, 8, 4, 4)
    assert item["neural"].shape == (2, 8, 4, 4)


def test_dataset_getitem_float32_dtype(tmp_path):
    p = str(tmp_path / "d.h5")
    _make_h5(p, n=3)
    ds = SyntheticH5Dataset(
        p, cache_cfg=_CACHE_CFG, dtype=torch.float32, layers=("layer_0", "layer_1")
    )
    item = ds[0]
    assert item["bold"].dtype == np.float32
    assert item["neural"].dtype == np.float32


def test_dataset_getitem_float16_dtype(tmp_path):
    p = str(tmp_path / "d.h5")
    _make_h5(p, n=3)
    ds = SyntheticH5Dataset(
        p, cache_cfg=_CACHE_CFG, dtype=torch.float16, layers=("layer_0", "layer_1")
    )
    item = ds[0]
    assert item["bold"].dtype == np.float16


# -------------------------
# SyntheticH5Dataset — meta / latents
# -------------------------

_LAYERS_2 = ("layer_0", "layer_1")


def test_dataset_return_meta_false_excludes_meta_keys(tmp_path):
    p = str(tmp_path / "d.h5")
    _make_h5(p, n=3, include_meta=True)
    ds = SyntheticH5Dataset(p, cache_cfg=_CACHE_CFG, return_meta=False, layers=_LAYERS_2)
    item = ds[0]
    for key in ("num_pulses", "source_layer", "source_position"):
        assert key not in item


def test_dataset_return_meta_true_includes_meta_keys(tmp_path):
    p = str(tmp_path / "d.h5")
    _make_h5(p, n=3, include_meta=True)
    ds = SyntheticH5Dataset(p, cache_cfg=_CACHE_CFG, return_meta=True, layers=_LAYERS_2)
    item = ds[0]
    assert "num_pulses" in item
    assert "source_layer" in item
    assert "source_position" in item
    assert isinstance(item["num_pulses"], int)
    assert isinstance(item["source_layer"], int)
    assert item["source_position"].ndim == 1  # per-sample 2-vector


def test_dataset_return_latents_true_includes_latent_keys(tmp_path):
    p = str(tmp_path / "d.h5")
    _make_h5(p, n=3, include_latents=True)
    ds = SyntheticH5Dataset(p, cache_cfg=_CACHE_CFG, return_latents=True, layers=_LAYERS_2)
    item = ds[0]
    for key in ("s", "f", "v", "q", "v_star", "q_star"):
        assert key in item, f"Missing latent key: {key}"


def test_dataset_return_latents_false_excludes_latent_keys(tmp_path):
    p = str(tmp_path / "d.h5")
    _make_h5(p, n=3, include_latents=True)
    ds = SyntheticH5Dataset(p, cache_cfg=_CACHE_CFG, return_latents=False, layers=_LAYERS_2)
    item = ds[0]
    for key in ("s", "f", "v", "q", "v_star", "q_star"):
        assert key not in item


def test_dataset_latent_shapes_match_expected(tmp_path):
    L, T, H, W, LT = 2, 8, 4, 4, 6
    p = str(tmp_path / "d.h5")
    _make_h5(p, n=3, t=T, h=H, w=W, layers=("layer_0", "layer_1"), latent_t=LT)
    ds = SyntheticH5Dataset(
        p, cache_cfg=_CACHE_CFG, return_latents=True, layers=("layer_0", "layer_1")
    )
    item = ds[0]
    for key in ("s", "f", "v", "q", "v_star", "q_star"):
        assert item[key].shape == (L, LT, H, W), f"Wrong shape for {key}: {item[key].shape}"


# -------------------------
# SyntheticH5Dataset — worker safety / pickling
# -------------------------


def test_dataset_getstate_nulls_all_handles(tmp_path):
    p = str(tmp_path / "d.h5")
    _make_h5(p, n=3)
    ds = SyntheticH5Dataset(p, cache_cfg=_CACHE_CFG, layers=_LAYERS_2)
    ds._ensure_open()
    assert ds._h5 is not None

    state = ds.__getstate__()
    assert state["_h5"] is None
    assert state["_bold_ds"] is None
    assert state["_x_ds"] is None
    assert state["_m_source_position"] is None


def test_dataset_pickle_roundtrip_closes_handles(tmp_path):
    p = str(tmp_path / "d.h5")
    _make_h5(p, n=3)
    ds = SyntheticH5Dataset(p, cache_cfg=_CACHE_CFG, layers=_LAYERS_2)
    ds._ensure_open()
    assert ds._h5 is not None

    ds2: SyntheticH5Dataset = pickle.loads(pickle.dumps(ds))
    assert ds2._h5 is None  # nulled by __getstate__

    # Can still read after re-opening lazily
    item = ds2[0]
    assert "bold" in item


def test_dataset_ensure_open_is_idempotent(tmp_path):
    p = str(tmp_path / "d.h5")
    _make_h5(p, n=3)
    ds = SyntheticH5Dataset(p, cache_cfg=_CACHE_CFG, layers=_LAYERS_2)
    ds._ensure_open()
    handle_first = ds._h5
    ds._ensure_open()
    assert ds._h5 is handle_first  # same object, not re-opened


def test_dataset_getitem_all_indices_accessible(tmp_path):
    """Every index in [0, N) should return a valid item without error."""
    n = 5
    p = str(tmp_path / "d.h5")
    _make_h5(p, n=n, layers=("layer_0",))
    ds = SyntheticH5Dataset(p, cache_cfg=_CACHE_CFG, layers=("layer_0",))
    for i in range(n):
        item = ds[i]
        assert item["bold"].shape[0] == 1  # 1 layer


# -------------------------
# SyntheticDataModule — setup and DataLoaders
# -------------------------

_DM_LAYERS = ("layer_0", "layer_1")
_DM_N = 10  # total samples in the fixture
_DM_TRAIN, _DM_VAL, _DM_TEST = 6, 2, 2


def _make_dm_h5(tmp_path, *, n: int = _DM_N, t: int = 8, h: int = 4, w: int = 4) -> str:
    """Write a minimal HDF5 fixture with meta and latents for DataModule tests."""
    path = str(tmp_path / "dm.h5")
    _make_h5(path, n=n, t=t, h=h, w=w, layers=_DM_LAYERS, include_meta=True, include_latents=True)
    return path


def _make_datamodule(h5_path: str) -> SyntheticDataModule:
    return SyntheticDataModule(
        data={
            "path": h5_path,
            "layers": list(_DM_LAYERS),
            "return_meta": True,
            "return_latents": True,
            "dtype": "float32",
        },
        split={
            "train_count": _DM_TRAIN,
            "val_count": _DM_VAL,
            "test_count": _DM_TEST,
            "seed": 42,
            "shuffle": True,
        },
        loader={"batch_size": 2, "num_workers": 0, "drop_last": True},
        h5_cache={},
    )


class TestSyntheticDataModule:
    def test_setup_creates_all_three_splits(self, tmp_path):
        """setup() populates ds_train, ds_val, ds_test."""
        dm = _make_datamodule(_make_dm_h5(tmp_path))
        dm.setup()
        assert dm.ds_train is not None
        assert dm.ds_val is not None
        assert dm.ds_test is not None

    def test_split_sizes_match_config(self, tmp_path):
        """Each split contains exactly the requested number of samples."""
        dm = _make_datamodule(_make_dm_h5(tmp_path))
        dm.setup()
        assert len(dm.ds_train) == _DM_TRAIN
        assert len(dm.ds_val) == _DM_VAL
        assert len(dm.ds_test) == _DM_TEST

    def test_train_val_test_indices_are_disjoint(self, tmp_path):
        """No sample index appears in more than one split."""
        dm = _make_datamodule(_make_dm_h5(tmp_path))
        dm.setup()
        train_idx = set(dm.ds_train.indices)
        val_idx = set(dm.ds_val.indices)
        test_idx = set(dm.ds_test.indices)
        assert train_idx.isdisjoint(val_idx), "train and val share indices"
        assert train_idx.isdisjoint(test_idx), "train and test share indices"
        assert val_idx.isdisjoint(test_idx), "val and test share indices"

    def test_train_loader_batch_has_all_required_keys(self, tmp_path):
        """A train batch contains bold, neural, source_position, meta, and latents."""
        dm = _make_datamodule(_make_dm_h5(tmp_path))
        dm.setup()
        loader = dm.train_dataloader()
        batch = next(iter(loader))
        required = {
            "bold",
            "neural",
            "source_position",
            "num_pulses",
            "source_layer",
            "s",
            "f",
            "v",
            "q",
            "v_star",
            "q_star",
        }
        assert required.issubset(batch.keys()), f"Missing keys: {required - batch.keys()}"

    def test_train_batch_bold_shape_is_correct(self, tmp_path):
        """bold tensor in a train batch has shape [batch_size, L, T, H, W]."""
        dm = _make_datamodule(_make_dm_h5(tmp_path))
        dm.setup()
        batch = next(iter(dm.train_dataloader()))
        assert batch["bold"].shape == (2, len(_DM_LAYERS), 8, 4, 4)

    def test_split_is_reproducible_with_same_seed(self, tmp_path):
        """Two DataModules with the same seed produce identical split indices."""
        h5_path = _make_dm_h5(tmp_path)
        dm1 = _make_datamodule(h5_path)
        dm2 = _make_datamodule(h5_path)
        dm1.setup()
        dm2.setup()
        assert dm1.ds_train.indices == dm2.ds_train.indices
        assert dm1.ds_val.indices == dm2.ds_val.indices

    def test_val_dataloader_batch_bold_shape_is_correct(self, tmp_path):
        """bold tensor in a val batch has the expected spatial shape."""
        dm = _make_datamodule(_make_dm_h5(tmp_path))
        dm.setup()
        batch = next(iter(dm.val_dataloader()))
        assert batch["bold"].shape == (2, len(_DM_LAYERS), 8, 4, 4)
