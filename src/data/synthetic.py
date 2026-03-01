from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pytorch_lightning as pl

try:
    from omegaconf import DictConfig
except Exception:  # pragma: no cover
    DictConfig = Any


def _torch_dtype(dtype_str: str) -> torch.dtype:
    s = str(dtype_str).lower()
    if s in ("float16", "fp16", "half"):
        return torch.float16
    if s in ("float32", "fp32", "float"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def _np_dtype(torch_dtype: torch.dtype) -> np.dtype:
    if torch_dtype == torch.float16:
        return np.float16
    if torch_dtype == torch.float32:
        return np.float32
    raise ValueError(f"Unsupported torch dtype: {torch_dtype}")


def _open_h5(path: str, cache_cfg: Mapping[str, Any]) -> h5py.File:
    """
    Notes:
    - swmr=False is typically faster if you are not writing during training.
    - rdcc_* controls HDF5 chunk cache per file handle (thus per worker).
    """
    return h5py.File(
        path,
        "r",
        libver="latest",
        swmr=bool(cache_cfg.get("swmr", False)),
        rdcc_nbytes=int(cache_cfg.get("rdcc_nbytes", 256 * 1024 * 1024)),
        rdcc_nslots=int(cache_cfg.get("rdcc_nslots", 200_003)),
        rdcc_w0=float(cache_cfg.get("rdcc_w0", 0.75)),
    )


def _compute_split_counts(n: int, split: Mapping[str, Any]) -> Tuple[int, int, int]:
    """
    Supports either explicit counts (train_count/val_count/test_count)
    OR fraction-based (train_frac/val_frac/test_frac). If both are present,
    counts take precedence when any count is provided.
    """
    train_count = split.get("train_count", None)
    val_count = split.get("val_count", None)
    test_count = split.get("test_count", None)

    # --- counts take precedence if any count is set ---
    if train_count is not None or val_count is not None or test_count is not None:
        tr = int(train_count or 0)
        va = int(val_count or 0)
        te = int(test_count or 0)
        if tr + va + te > n:
            raise ValueError(f"Split counts exceed dataset size: {tr}+{va}+{te} > {n}")
        # If train_count missing, allocate remainder to train.
        if split.get("train_count", None) is None:
            tr = n - (va + te)
        return tr, va, te

    # --- fraction-based ---
    # allow either train_frac+val_frac+test_frac or just val/test
    tf = split.get("test_frac", 0.1)
    vf = split.get("val_frac", 0.1)
    trf = split.get("train_frac", None)

    vf = float(vf)
    tf = float(tf)

    if vf < 0 or tf < 0:
        raise ValueError("val_frac/test_frac must be non-negative")

    if trf is None:
        if vf + tf > 1.0 + 1e-8:
            raise ValueError("val_frac + test_frac must be <= 1.0")
        va = int(round(n * vf))
        te = int(round(n * tf))
        tr = n - va - te
        return tr, va, te

    trf = float(trf)
    if trf < 0 or trf > 1:
        raise ValueError("train_frac must be in [0,1]")
    if abs((trf + vf + tf) - 1.0) > 1e-6:
        # Don’t silently renormalize; make it explicit.
        raise ValueError(f"train_frac+val_frac+test_frac must sum to 1. Got {trf+vf+tf}")
    tr = int(round(n * trf))
    va = int(round(n * vf))
    te = n - tr - va  # remainder to test to keep sum exact
    return tr, va, te


class SyntheticH5Dataset(Dataset):
    """
    Reads:
      /layer_k/bold : (N, T, H, W) ideally chunked (1,T,H,W)
      /layer_k/x    : (N, T, H, W) ideally chunked (1,T,H,W)

    Returns tensors:
      bold  : (L, T, H, W)
      neural: (L, T, H, W)

    Performance:
    - each DataLoader worker opens its own file handle lazily on first __getitem__
    - dataset handles are cached to avoid repeated HDF5 tree traversal
    - read_direct eliminates intermediate allocations on every sample read
    - np_dtype resolved once at construction
    """

    def __init__(
        self,
        path: str,
        *,
        cache_cfg: Mapping[str, Any],
        layers: Sequence[str] = ("layer_0", "layer_1", "layer_2"),
        dtype: torch.dtype = torch.float32,
        return_meta: bool = False,
    ):
        super().__init__()
        self.path = str(path)
        self.layers = tuple(layers)
        self.dtype = dtype
        self._np_dtype = _np_dtype(dtype)  # resolved once, reused every __getitem__
        self.return_meta = bool(return_meta)
        self.cache_cfg = dict(cache_cfg)

        # worker-local handles — never pickled
        self._h5: Optional[h5py.File] = None
        self._bold_ds: Optional[list[h5py.Dataset]] = None
        self._x_ds: Optional[list[h5py.Dataset]] = None
        self._m_num_pulses: Optional[h5py.Dataset] = None
        self._m_source_layer: Optional[h5py.Dataset] = None
        self._m_source_position: Optional[h5py.Dataset] = None

        # read static shape info once in the main process
        with h5py.File(self.path, "r") as f:
            self.N = int(f[self.layers[0]]["bold"].shape[0])
            t, h, w = f[self.layers[0]]["bold"].shape[1:]
            self._window_shape = (int(t), int(h), int(w))

    def __len__(self) -> int:
        return self.N

    def __getstate__(self) -> dict:
        # null all file handles so workers always start fresh
        d = dict(self.__dict__)
        for k in ("_h5", "_bold_ds", "_x_ds",
                  "_m_num_pulses", "_m_source_layer", "_m_source_position"):
            d[k] = None
        return d

    def __del__(self) -> None:
        try:
            if self._h5 is not None:
                self._h5.close()
                self._h5 = None
        except ImportError:
            pass  # interpreter shutting down, h5py already unloaded

    def _ensure_open(self) -> None:
        if self._h5 is not None:
            return
        self._h5 = _open_h5(self.path, self.cache_cfg)
        self._bold_ds = [self._h5[lyr]["bold"] for lyr in self.layers]
        self._x_ds = [self._h5[lyr]["x"] for lyr in self.layers]
        if self.return_meta:
            self._m_num_pulses = self._h5["meta"]["num_pulses"]
            self._m_source_layer = self._h5["meta"]["source_layer"]
            self._m_source_position = self._h5["meta"]["source_position"]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        self._ensure_open()

        L = len(self.layers)
        T, H, W = self._window_shape

        bold = np.empty((L, T, H, W), dtype=self._np_dtype)
        x = np.empty((L, T, H, W), dtype=self._np_dtype)

        for l in range(L):
            self._bold_ds[l].read_direct(bold, source_sel=np.s_[idx], dest_sel=np.s_[l])
            self._x_ds[l].read_direct(x, source_sel=np.s_[idx], dest_sel=np.s_[l])

        out: Dict[str, Any] = {"bold": bold, "neural": x}

        if self.return_meta:
            out.update({
                "num_pulses": int(self._m_num_pulses[idx]),
                "source_layer": int(self._m_source_layer[idx]),
                "source_position": self._m_source_position[idx],
            })

        return out
    

class SyntheticDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data: Mapping[str, Any],
        split: Mapping[str, Any],
        loader: Mapping[str, Any],
        h5_cache: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.data_config = dict(data)
        self.split_config = dict(split)
        self.loader_config = dict(loader)
        self.h5_cache_config = dict(h5_cache)

        self.dataset_full: Optional[SyntheticH5Dataset] = None
        self.ds_train: Optional[Subset] = None
        self.ds_val: Optional[Subset] = None
        self.ds_test: Optional[Subset] = None

    def setup(self, stage: str | None = None) -> None:
        data_path = self.data_config.get("path")
        if data_path is None:
            raise ValueError("data.path must be set")

        layers_val = self.data_config.get("layers", ("layer_0", "layer_1", "layer_2"))
        layers = tuple(layers_val) if layers_val is not None else ("layer_0", "layer_1", "layer_2")

        dtype = _torch_dtype(self.data_config.get("dtype", "float32"))
        return_meta = bool(self.data_config.get("return_meta", False))

        self.dataset_full = SyntheticH5Dataset(
            path=str(data_path),
            layers=layers,
            dtype=dtype,
            return_meta=return_meta,
            cache_cfg=self.h5_cache_config,
        )

        n = len(self.dataset_full)
        n_train, n_val, n_test = _compute_split_counts(n, self.split_config)

        seed = int(self.split_config.get("seed", 42))
        shuffle = bool(self.split_config.get("shuffle", True))

        g = torch.Generator().manual_seed(seed)
        indices = torch.arange(n)
        if shuffle:
            indices = indices[torch.randperm(n, generator=g)]

        train_idx = indices[:n_train].tolist()
        val_idx = indices[n_train : n_train + n_val].tolist()
        test_idx = indices[n_train + n_val : n_train + n_val + n_test].tolist()

        self.ds_train = Subset(self.dataset_full, train_idx)
        self.ds_val = Subset(self.dataset_full, val_idx)
        self.ds_test = Subset(self.dataset_full, test_idx)

    def _make_loader(self, ds: Subset, *, shuffle: bool, drop_last: bool) -> DataLoader:
        bs = int(self.loader_config.get("batch_size", 2))
        num_workers = int(self.loader_config.get("num_workers", 0))

        pin_memory = bool(self.loader_config.get("pin_memory", False))
        persistent_workers = bool(self.loader_config.get("persistent_workers", False)) and num_workers > 0

        # Only valid when num_workers > 0
        prefetch_factor = self.loader_config.get("prefetch_factor", 2)
        prefetch_factor = int(prefetch_factor) if (num_workers > 0 and prefetch_factor is not None) else None

        return DataLoader(
            ds,
            batch_size=bs,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

    def train_dataloader(self) -> DataLoader:
        assert self.ds_train is not None, "train dataset not initialized"
        return self._make_loader(
            self.ds_train,
            shuffle=True,
            drop_last=bool(self.loader_config.get("drop_last", True)),
        )

    def val_dataloader(self) -> DataLoader:
        assert self.ds_val is not None, "val dataset not initialized"
        return self._make_loader(self.ds_val, shuffle=False, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        assert self.ds_test is not None, "test dataset not initialized"
        return self._make_loader(self.ds_test, shuffle=False, drop_last=False)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()