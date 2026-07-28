"""HDF5-backed dataset/datamodule for training on `run_sim.py`-generated simulations.

HDF5 layout (see `SyntheticH5Dataset`'s docstring for per-layer dataset shapes):
one `/layer_k/` group per cortical layer (bold, x, and -- if latents were saved --
s, f, v, q[, v_star, q_star]), plus a `/meta` group with per-sample source
metadata and the exact simulation config as a JSON string attribute (read by
`SyntheticDataModule.sim_config`).

Worker-local file handles: `h5py.File` handles aren't safely shareable across a
`DataLoader` worker-process fork, so `SyntheticH5Dataset` opens its file lazily,
per-process, via `_ensure_open` (called at the top of every `__getitem__`) rather
than in `__init__`, and `__getstate__` nulls out any handles already open in the
main process before a worker is forked, so each worker starts with none and opens
its own on first access.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from omegaconf import DictConfig
except Exception:
    DictConfig = Any


def _torch_dtype(dtype_str: str) -> torch.dtype:
    """Config string ("float16"/"fp16"/"half", "float32"/"fp32"/"float",
    case-insensitive) -> torch dtype.

    Raises:
        ValueError: If `dtype_str` doesn't match one of the names above.
    """
    s = str(dtype_str).lower()
    if s in ("float16", "fp16", "half"):
        return torch.float16
    if s in ("float32", "fp32", "float"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def _np_dtype(torch_dtype: torch.dtype) -> np.dtype:
    """`_torch_dtype`'s inverse, restricted to the two dtypes it can produce.

    Raises:
        ValueError: If `torch_dtype` isn't `torch.float16` or `torch.float32`.
    """
    if torch_dtype == torch.float16:
        return np.float16
    if torch_dtype == torch.float32:
        return np.float32
    raise ValueError(f"Unsupported torch dtype: {torch_dtype}")


def discover_layers(path: str) -> Tuple[str, ...]:
    """Auto-discover layer_* groups in a simulation HDF5 file, sorted by index."""
    with h5py.File(str(path), "r") as f:
        names = sorted(
            (k for k in f.keys() if k.startswith("layer_")),
            key=lambda s: int(s.split("_")[1]),
        )
    if not names:
        raise ValueError(f"No layer_* groups found in {path}")
    return tuple(names)


def _open_h5(path: str, cache_cfg: Mapping[str, Any]) -> h5py.File:
    """Open `path` read-only with per-handle HDF5 chunk-cache settings from
    `cache_cfg` (each `DataLoader` worker gets its own handle and thus its own
    cache -- see `SyntheticH5Dataset`'s worker-local-handle note).

    Args:
        cache_cfg: `swmr` (default False -- typically faster if nothing is
            writing to the file during training), `rdcc_nbytes` (default
            256 MiB), `rdcc_nslots` (default 200_003), `rdcc_w0` (default
            0.75). See h5py's `File` docs for what each controls.
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


def compute_split_counts(n: int, split: Mapping[str, Any]) -> Tuple[int, int, int]:
    """(train, val, test) example counts summing to `n`.

    Supports either explicit counts (train_count/val_count/test_count) OR
    fraction-based (train_frac/val_frac/test_frac). If both are present,
    counts take precedence when any count is provided. In the count-based
    path, an unset `train_count` gets the remainder (`n - val - test`); in the
    fraction-based path, an unset `train_frac` gets the remainder fraction
    (val/test are rounded independently and train absorbs the rounding
    error), while a given `train_frac` requires all three to sum to 1
    (rounding errors go entirely to test, the remainder term).

    Args:
        n: Total example count to split.
        split: See above -- `test_frac`/`val_frac` default to 0.1 each in the
            fraction-based path if unset.

    Raises:
        ValueError: If the counts exceed `n`; if `val_frac`/`test_frac` are
            negative; if `train_frac` is unset and `val_frac + test_frac >
            1`; or if `train_frac` is set and the three fractions don't sum
            to 1 (within 1e-6).
    """
    train_count = split.get("train_count", None)
    val_count = split.get("val_count", None)
    test_count = split.get("test_count", None)

    # counts take precedence if any count is set
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
        raise ValueError(f"train_frac+val_frac+test_frac must sum to 1. Got {trf + vf + tf}")
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
    """

    def __init__(
        self,
        path: str,
        *,
        cache_cfg: Mapping[str, Any],
        layers: Sequence[str] = ("layer_0",),
        dtype: torch.dtype = torch.float32,
        return_meta: bool = False,
        return_latents: bool = False,
    ):
        """
        Args:
            path: HDF5 simulation file (see module docstring for layout).
            cache_cfg: Forwarded to `_open_h5` for every worker's file handle.
            layers: Which `/layer_k/` groups to read, in the order stacked
                into this dataset's layer axis.
            dtype: Output tensor dtype for bold/neural (and latents, if
                requested); the HDF5 arrays are read directly into a numpy
                buffer of the matching numpy dtype (`_np_dtype(dtype)`), not
                cast after the fact.
            return_meta: If True, also return per-sample source metadata.
            return_latents: If True, also return s/f/v/q(/v_star/q_star)
                ground-truth latents.

        Note:
            Latents can have a different recorded length than bold/x
            (`self.lt` vs the bold/x time length in `self._window_shape`) --
            `__getitem__` allocates each accordingly, not assuming they match.
        """
        super().__init__()
        self.path = str(path)
        self.layers = tuple(layers)
        self.dtype = dtype
        self._np_dtype = _np_dtype(dtype)  # resolved once, reused every __getitem__
        self.return_meta = bool(return_meta)
        self.return_latents = bool(return_latents)
        self.cache_cfg = dict(cache_cfg)

        # worker-local handles
        self._h5: Optional[h5py.File] = None
        self._bold_ds: Optional[list[h5py.Dataset]] = None
        self._x_ds: Optional[list[h5py.Dataset]] = None
        self._m_num_sources: Optional[h5py.Dataset] = None
        self._m_source_layer: Optional[h5py.Dataset] = None
        self._m_source_position: Optional[h5py.Dataset] = None
        self._m_source_num_pulses: Optional[h5py.Dataset] = None
        self._m_latent_s: Optional[list[h5py.Dataset]] = None
        self._m_latent_f: Optional[list[h5py.Dataset]] = None
        self._m_latent_v: Optional[list[h5py.Dataset]] = None
        self._m_latent_q: Optional[list[h5py.Dataset]] = None
        self._m_latent_v_star: Optional[list[h5py.Dataset]] = None
        self._m_latent_q_star: Optional[list[h5py.Dataset]] = None

        # read static shape info once in the main process
        with h5py.File(self.path, "r") as f:
            self.N = int(f[self.layers[0]]["bold"].shape[0])
            t, h, w = f[self.layers[0]]["bold"].shape[1:]
            self.lt = f[self.layers[0]]["s"].shape[1] if self.return_latents else t
            self._window_shape = (int(t), int(h), int(w))

    def __len__(self) -> int:
        return self.N

    def __getstate__(self) -> dict:
        """Pickling support: strip every open HDF5 handle before this instance
        is sent to a `DataLoader` worker process, so each worker calls
        `_ensure_open` itself on first access instead of inheriting (or trying
        to pickle) the main process's handles. See the module docstring's
        *Worker-local file handles* note."""
        # null all file handles so workers always start fresh
        d = dict(self.__dict__)
        for k in (
            "_h5",
            "_bold_ds",
            "_x_ds",
            "_m_num_sources",
            "_m_source_layer",
            "_m_source_position",
            "_m_source_num_pulses",
            "_m_latent_s",
            "_m_latent_f",
            "_m_latent_v",
            "_m_latent_q",
            "_m_latent_v_star",
            "_m_latent_q_star",
        ):
            d[k] = None
        return d

    def __del__(self) -> None:
        """Close this instance's own HDF5 handle, if it opened one.

        Warning:
            `ImportError` during close is silently swallowed -- at interpreter
            shutdown, `h5py`'s own internals may already be unloaded, and a
            `__del__` that raises there produces a spurious, unhelpful
            "exception ignored in __del__" warning rather than a real error.
        """
        try:
            if self._h5 is not None:
                self._h5.close()
                self._h5 = None
        except ImportError:
            pass  # interpreter shutting down, h5py already unloaded

    def _ensure_open(self) -> None:
        """Open this process's own HDF5 handle and per-layer dataset refs, if
        not already open (idempotent no-op otherwise). See the module
        docstring's *Worker-local file handles* note for why this exists
        instead of opening in `__init__`.

        Note:
            v_star/q_star are only opened if `len(self.layers) > 1` (matching
            the drain-mode-is-multi-layer convention elsewhere) -- a
            single-layer dataset never reads them even if `return_latents`
            and the file happens to have them.
        """
        if self._h5 is not None:
            return
        self._h5 = _open_h5(self.path, self.cache_cfg)
        self._bold_ds = [self._h5[lyr]["bold"] for lyr in self.layers]
        self._x_ds = [self._h5[lyr]["x"] for lyr in self.layers]
        if self.return_meta:
            self._m_num_sources = self._h5["meta"]["num_sources"]
            self._m_source_layer = self._h5["meta"]["sources"]["layer"]
            self._m_source_position = self._h5["meta"]["sources"]["position"]
            self._m_source_num_pulses = self._h5["meta"]["sources"]["num_pulses"]
        if self.return_latents:
            self._m_latent_s = [self._h5[lyr]["s"] for lyr in self.layers]
            self._m_latent_f = [self._h5[lyr]["f"] for lyr in self.layers]
            self._m_latent_v = [self._h5[lyr]["v"] for lyr in self.layers]
            self._m_latent_q = [self._h5[lyr]["q"] for lyr in self.layers]
            if len(self.layers) > 1:
                self._m_latent_v_star = [self._h5[lyr]["v_star"] for lyr in self.layers]
                self._m_latent_q_star = [self._h5[lyr]["q_star"] for lyr in self.layers]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """One sample, read directly from HDF5 into pre-allocated numpy buffers
        (`Dataset.read_direct`, one layer at a time -- avoids an intermediate
        per-layer array before stacking).

        Returns:
            Always: "bold", "neural" ([L, T, H, W], `self.dtype`).
            If `return_meta`: "num_sources" (int), "source_layer" ([max_sources]),
            "source_position" ([max_sources, 2]), "num_pulses" ([max_sources])
            -- only the first `num_sources` entries of each padded array are
            valid; the rest are whatever `run_sim.py` padded them with.
            If `return_latents`: "s", "f", "v", "q" ([L, self.lt, H, W]), plus
            "v_star", "q_star" (same shape) if `len(self.layers) > 1`.
        """
        self._ensure_open()

        L = len(self.layers)
        T, H, W = self._window_shape

        bold = np.empty((L, T, H, W), dtype=self._np_dtype)
        x = np.empty((L, T, H, W), dtype=self._np_dtype)
        for layer_index in range(L):
            self._bold_ds[layer_index].read_direct(
                bold, source_sel=np.s_[idx], dest_sel=np.s_[layer_index]
            )
            self._x_ds[layer_index].read_direct(
                x, source_sel=np.s_[idx], dest_sel=np.s_[layer_index]
            )

        out: Dict[str, Any] = {"bold": bold, "neural": x}

        if self.return_meta:
            out.update(
                {
                    # padded to max_sources; only the first `num_sources` entries are valid
                    "num_sources": int(self._m_num_sources[idx]),
                    "source_layer": self._m_source_layer[idx],  # [max_sources]
                    "source_position": self._m_source_position[idx],  # [max_sources, 2]
                    "num_pulses": self._m_source_num_pulses[idx],  # [max_sources]
                }
            )

        if self.return_latents:
            s = np.empty((L, self.lt, H, W), dtype=self._np_dtype)
            f = np.empty((L, self.lt, H, W), dtype=self._np_dtype)
            v = np.empty((L, self.lt, H, W), dtype=self._np_dtype)
            q = np.empty((L, self.lt, H, W), dtype=self._np_dtype)

            for layer_index in range(L):
                self._m_latent_s[layer_index].read_direct(
                    s, source_sel=np.s_[idx], dest_sel=np.s_[layer_index]
                )
                self._m_latent_f[layer_index].read_direct(
                    f, source_sel=np.s_[idx], dest_sel=np.s_[layer_index]
                )
                self._m_latent_v[layer_index].read_direct(
                    v, source_sel=np.s_[idx], dest_sel=np.s_[layer_index]
                )
                self._m_latent_q[layer_index].read_direct(
                    q, source_sel=np.s_[idx], dest_sel=np.s_[layer_index]
                )

            out.update({"s": s, "f": f, "v": v, "q": q})

            if self._m_latent_v_star is not None:
                v_star = np.empty((L, self.lt, H, W), dtype=self._np_dtype)
                q_star = np.empty((L, self.lt, H, W), dtype=self._np_dtype)
                for layer_index in range(L):
                    self._m_latent_v_star[layer_index].read_direct(
                        v_star, source_sel=np.s_[idx], dest_sel=np.s_[layer_index]
                    )
                    self._m_latent_q_star[layer_index].read_direct(
                        q_star, source_sel=np.s_[idx], dest_sel=np.s_[layer_index]
                    )
                out.update({"v_star": v_star, "q_star": q_star})

        return out


class SyntheticDataModule(pl.LightningDataModule):
    """Splits one HDF5 simulation file into train/val/test `SyntheticH5Dataset`
    subsets, and builds their `DataLoader`s.

    Each split gets its own `SyntheticH5Dataset` instance (see `_make_dataset`),
    not a shared one -- so `data`'s per-split override keys ("train"/"val"/"test",
    see `_resolve_split_cfg`) can give each split different `return_meta`/
    `return_latents`/`dtype`/etc, e.g. only requesting latents for validation.
    """

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

        self.ds_train: Optional[Subset] = None
        self.ds_val: Optional[Subset] = None
        self.ds_test: Optional[Subset] = None

    def _resolve_split_cfg(self, split_name: str) -> dict:
        """Merge base data_config with any per-split overrides under data_config[split_name]."""
        cfg = dict(self.data_config)
        overrides = cfg.pop(split_name, None)
        if overrides:
            cfg.update(overrides)
        # also strip the other split override keys so they don't pollute downstream
        for key in ("train", "val", "test"):
            cfg.pop(key, None)
        return cfg

    def _make_dataset(self, cfg: dict) -> SyntheticH5Dataset:
        """Build a `SyntheticH5Dataset` from a resolved (post-`_resolve_split_cfg`)
        config dict.

        Args:
            cfg: Must have "path"; "layers" of "auto"/None (default) discovers
                every `layer_*` group via `discover_layers` instead of taking
                an explicit list. "dtype"/"return_meta"/"return_latents" as in
                `SyntheticH5Dataset.__init__` (string dtype, resolved here).

        Raises:
            ValueError: If `cfg["path"]` is missing.
        """
        data_path = cfg.get("path")
        if data_path is None:
            raise ValueError("data.path must be set")
        layers_val = cfg.get("layers", "auto")
        if layers_val is None or layers_val == "auto":
            layers = discover_layers(data_path)
        else:
            layers = tuple(layers_val)
        return SyntheticH5Dataset(
            path=str(data_path),
            layers=layers,
            dtype=_torch_dtype(cfg.get("dtype", "float32")),
            return_meta=bool(cfg.get("return_meta", False)),
            return_latents=bool(cfg.get("return_latents", False)),
            cache_cfg=self.h5_cache_config,
        )

    def setup(self, stage: str | None = None) -> None:
        """Build `ds_train`/`ds_val`/`ds_test` as disjoint, reproducibly-seeded
        `Subset`s of the total sample count.

        A throwaway dataset (built from the base, non-split-specific config)
        is opened only to read its length; a fresh dataset is then built per
        split (via `_make_dataset(self._resolve_split_cfg(<split>))`), so each
        split can have its own `return_meta`/`return_latents`/etc, and none of
        the three share an open file handle. Indices are assigned contiguously
        after an optional `torch.Generator`-seeded shuffle: `[0:n_train)` ->
        train, `[n_train:n_train+n_val)` -> val, the rest (up to `n_test`) ->
        test -- so train/val/test are always disjoint regardless of shuffling.
        """
        # Use a temporary dataset to get N and compute split indices
        base_cfg = self._resolve_split_cfg("__none__")
        _tmp = self._make_dataset(base_cfg)
        n = len(_tmp)

        n_train, n_val, n_test = compute_split_counts(n, self.split_config)

        seed = int(self.split_config.get("seed", 42))
        shuffle = bool(self.split_config.get("shuffle", True))

        g = torch.Generator().manual_seed(seed)
        indices = torch.arange(n)
        if shuffle:
            indices = indices[torch.randperm(n, generator=g)]

        train_idx = indices[:n_train].tolist()
        val_idx = indices[n_train : n_train + n_val].tolist()
        test_idx = indices[n_train + n_val : n_train + n_val + n_test].tolist()

        self.ds_train = Subset(self._make_dataset(self._resolve_split_cfg("train")), train_idx)
        self.ds_val = Subset(self._make_dataset(self._resolve_split_cfg("val")), val_idx)
        self.ds_test = Subset(self._make_dataset(self._resolve_split_cfg("test")), test_idx)

    def _make_loader(self, ds: Subset, *, shuffle: bool, drop_last: bool) -> DataLoader:
        """Build a `DataLoader` over `ds` from `self.loader_config` (batch_size
        default 2, num_workers default 0, pin_memory default False,
        persistent_workers default False -- forced False if num_workers=0,
        prefetch_factor default 2 -- forced None if num_workers=0, since torch
        rejects a non-None prefetch_factor with no workers)."""
        bs = int(self.loader_config.get("batch_size", 2))
        num_workers = int(self.loader_config.get("num_workers", 0))

        pin_memory = bool(self.loader_config.get("pin_memory", False))
        persistent_workers = (
            bool(self.loader_config.get("persistent_workers", False)) and num_workers > 0
        )

        # Only valid when num_workers > 0
        prefetch_factor = self.loader_config.get("prefetch_factor", 2)
        prefetch_factor = (
            int(prefetch_factor) if (num_workers > 0 and prefetch_factor is not None) else None
        )

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
        return self._make_loader(self.ds_val, shuffle=False, drop_last=True)

    def test_dataloader(self) -> DataLoader:
        assert self.ds_test is not None, "test dataset not initialized"
        return self._make_loader(self.ds_test, shuffle=False, drop_last=True)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    @property
    def sim_config(self) -> dict:
        """The exact `run_sim.py` config that produced this datamodule's HDF5
        file, read fresh from `/meta`'s "config" JSON attribute on every
        access (not cached).

        Raises:
            ValueError: If `self.data_config["path"]` is unset.
        """
        import json

        path = self.data_config.get("path")
        if path is None:
            raise ValueError("data.path is not set in datamodule config")
        with h5py.File(str(path), "r") as f:
            return json.loads(f["meta"].attrs["config"])
