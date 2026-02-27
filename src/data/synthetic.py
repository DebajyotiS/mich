# synthetic.py
# HDF5 (single-file) loader for layered BOLD + x with full-window reads.
# Includes: loading utilities, Dataset, (un)preprocessing placeholders, and a LightningDataModule
# with train/val/test splits driven by Hydra config.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, Subset
import pytorch_lightning as pl

try:
    from omegaconf import DictConfig
except Exception:  # pragma: no cover
    DictConfig = Any


def _torch_dtype(dtype_str: str) -> torch.dtype:
    s = dtype_str.lower()
    if s in ("float16", "fp16", "half"):
        return torch.float16
    if s in ("float32", "fp32", "float"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def _open_h5(
    path: str,
    cache_cfg: dict,
) -> h5py.File:
    # swmr=True is safe even if you don't write during training
    return h5py.File(
        path,
        "r",
        libver="latest",
        swmr=True,
        rdcc_nbytes=cache_cfg["rdcc_nbytes"],
        rdcc_nslots=cache_cfg["rdcc_nslots"],
        rdcc_w0=cache_cfg["rdcc_w0"],
    )



def _compute_split_counts(n: int, split: Mapping[str, Any]) -> Tuple[int, int, int]:
    train_count = split.get("train_count", None)
    val_count = split.get("val_count", None)
    test_count = split.get("test_count", None)
    # Fractions take precedence if counts not fully specified.
    if train_count is not None or val_count is not None or test_count is not None:
        tr = train_count or 0
        va = val_count or 0
        te = test_count or 0
        if tr + va + te > n:
            raise ValueError(f"Split counts exceed dataset size: {tr}+{va}+{te} > {n}")
        # allocate remainder to train if train_count is missing
        if split.get("train_count", None) is None:
            tr = n - (va + te)
        return tr, va, te

    # fraction-based
    val_frac = split.get("val_frac", 0.1)
    test_frac = split.get("test_frac", 0.1)
    vf = float(val_frac)
    tf = float(test_frac)
    if vf < 0 or tf < 0:
        raise ValueError("val_frac/test_frac must be non-negative")
    if vf + tf > 1.0 + 1e-8:
        raise ValueError("val_frac + test_frac must be <= 1.0")
    va = int(round(n * vf))
    te = int(round(n * tf))
    tr = n - va - te
    return tr, va, te



class SyntheticH5Dataset(Dataset):
    """
    Reads only:
    /layer_k/bold : (N, T, H, W) with chunks (1,T,H,W)
    /layer_k/x    : (N, T, H, W) with chunks (1,T,H,W)
    Stacks over layers -> (L, T, H, W).

    Designed for full-window access and minimal overhead.
    Each DataLoader worker opens its own file handle lazily and reuses it.
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
        self.path = path
        self.layers = tuple(layers)
        self.dtype = dtype
        self.return_meta = bool(return_meta)
        self.cache_cfg = cache_cfg

        self._h5: Optional[h5py.File] = None

        with h5py.File(self.path, "r") as f:
            self.N = int(f[self.layers[0]]["bold"].shape[0])

    def __len__(self) -> int:
        return self.N

    def __getstate__(self):
        # do not pickle open file handles into workers
        d = dict(self.__dict__)
        d["_h5"] = None
        return d

    def __del__(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def _f(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = _open_h5(self.path, self.cache_cfg)
        return self._h5

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        f = self._f()

        bold = np.stack([f[lyr]["bold"][idx] for lyr in self.layers], axis=0)  # (L,T,H,W)
        x = np.stack([f[lyr]["x"][idx] for lyr in self.layers], axis=0)        # (L,T,H,W)

        out = {
            "bold": bold.astype(np.float32),
            "neural": x.astype(np.float32),
        }
        if self.return_meta:
            num_pulses = f["meta"]["num_pulses"][idx]
            source_layer = f["meta"]["source_layer"][idx]
            source_position = f["meta"]["source_position"][idx]
            out.update({
                "num_pulses": int(num_pulses),
                "source_layer": int(source_layer),
                "source_position": source_position,
            })
        return out


class SyntheticDataModule(pl.LightningDataModule):
    
    def __init__(self, data: Mapping[str, Any], split: Mapping[str, Any], loader: Mapping[str, Any], h5_cache: Mapping[str, Any]) -> None:
        super().__init__()
        self.data_config = data
        self.split_config = split
        self.loader_config = loader
        self.h5_cache_config = h5_cache
    
    def setup(self, stage: str|None) -> None:
        data_path = self.data_config.get("path")
        layers_val = self.data_config.get("layers", ("layer_0", "layer_1", "layer_2"))
        layers = tuple(layers_val) if layers_val is not None else ("layer_0", "layer_1", "layer_2")
        dtype = _torch_dtype(self.data_config.get("dtype", "float16"))
        return_meta = bool(self.data_config.get("return_meta", False))
        
        self.dataset_full = SyntheticH5Dataset(
            path=data_path,
            layers=layers,
            dtype=dtype,
            return_meta=return_meta,
            cache_cfg=self.h5_cache_config,
        )
        
        n = len(self.dataset_full)
        n_train, n_val, n_test = _compute_split_counts(n, self.split_config)
        seed = self.split_config.get("seed", 42)
        g = torch.Generator().manual_seed(seed)
        indices = torch.arange(n)
        if self.split_config.get("shuffle", True):
            indices = indices[torch.randperm(n, generator=g)]

        train_idx = indices[:n_train].tolist()
        val_idx = indices[n_train:n_train + n_val].tolist()
        test_idx = indices[n_train + n_val:n_train + n_val + n_test].tolist()
        
        self.ds_train = Subset(self.dataset_full, train_idx)
        self.ds_val = Subset(self.dataset_full, val_idx)
        self.ds_test = Subset(self.dataset_full, test_idx)
    
    def train_dataloader(self) -> DataLoader:
        assert self.ds_train is not None, "train dataset not initialized"
        return DataLoader(
            self.ds_train,
            batch_size=self.loader_config.get("batch_size", 2),
            shuffle=True,
            drop_last=self.loader_config.get("drop_last", True),
            num_workers=self.loader_config.get("num_workers", 0),
            prefetch_factor=self.loader_config.get("prefetch_factor", 2),
        )
        
    def val_dataloader(self) -> DataLoader:
        assert self.ds_val is not None, "val dataset not initialized"
        return DataLoader(
            self.ds_val,
            batch_size=self.loader_config.get("batch_size", 2),
            shuffle=False,
            drop_last=False,
            num_workers=self.loader_config.get("num_workers", 0),
            prefetch_factor=self.loader_config.get("prefetch_factor", 2),
        )
    
    def test_dataloader(self) -> DataLoader:
        assert self.ds_test is not None, "test dataset not initialized"
        return DataLoader(
            self.ds_test,
            batch_size=self.loader_config.get("batch_size", 2),
            shuffle=False,
            drop_last=False,
            num_workers=self.loader_config.get("num_workers", 0),
            prefetch_factor=self.loader_config.get("prefetch_factor", 2),
        )
    
    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()