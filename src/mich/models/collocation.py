"""Collocation-point sampling and index-based gathering into [B, ..., L, T, H, W] tensors.

Cortical-layer index convention (shared with `mich.utils.plotting` and
`mich.models.mich_logging`, which must agree with `_layer_index` below rather than
re-deriving it): layer channel index 0=deep, 1=middle, 2=superficial -- increasing
index means increasing distance from white matter.

Dense/uniform sampling vocabulary, used throughout `_sample_collocation_indices*`:
a draw of `n_space` spatial points x `n_times` timesteps per (batch, layer) is split
into a "dense" share, biased spatially within `dense_spatial_radius` of a known
source position (round-robinned across sources so each gets ~equal coverage) and
temporally into the `[dense_time_lo, dense_time_hi]` window (as a fraction of T), and
a "uniform" share drawn independently and uniformly over the whole grid/timeline
(time uniform draw still respects `uniform_time_lo`). `dense_spatial_frac` and
`dense_time_frac` set each share's size. This concentrates collocation coverage
where the physics loss matters most (near the source, where sudden transients
happen) while still sampling the full domain.

`full_grid=True` (see `_full_grid_collocation_batch`) bypasses all of the above:
every `(t, h, w)` in the volume is used instead of a sample, and every other
sampling knob (`n_times`, `n_space`, `dense_*`, `uniform_time_lo`,
`source_position`/`num_sources`) is ignored rather than validated.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mich.models.blocks import HeinzleSignal


@dataclass(frozen=True)
class CollocationBatch:
    """Index tensors describing collocation points.

    t: [1, 1, n_times, n_space]   -- shared across batch and layers
    h: [B, 1, n_times, n_space]   -- per-sample spatial points
    w: [B, 1, n_times, n_space]
    """

    t: torch.Tensor
    h: torch.Tensor
    w: torch.Tensor


class CollocationMixin:
    """Signal/layer index lookups, collocation-point sampling, and index-gather helpers.

    Stateless (all methods are static) -- mixed into MICH purely to keep this cohesive
    block of tensor-indexing utilities out of the main model file.
    """

    @staticmethod
    def _make_time_grid(B: int, T: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Build the [0, 1]-normalised time grid the decoder is queried at.

        Returns:
            [B, T] tensor, `linspace(0, 1, T)` broadcast identically to every
            batch row (all samples share the same T timesteps).
        """
        return torch.linspace(0.0, 1.0, T, device=device, dtype=dtype).view(1, T).expand(B, T)

    @staticmethod
    def _signal_index(signal: HeinzleSignal | int) -> int:
        """Resolve a Heinzle signal name or channel index to its canonical index.

        Args:
            signal: One of `HeinzleSignal` ("x","s","f","v","q","vstar","qstar")
                or an already-resolved int index in [0, 6].

        Raises:
            IndexError: If `signal` is an int outside [0, 6].
            KeyError: If `signal` is a string not in the mapping.
        """
        mapping = {"x": 0, "s": 1, "f": 2, "v": 3, "q": 4, "vstar": 5, "qstar": 6}
        if isinstance(signal, int):
            if 0 <= signal < 7:
                return signal
            raise IndexError(f"signal index must be in [0,6], got {signal}")
        return mapping[signal]

    @staticmethod
    def _layer_index(layer: str) -> int:
        """Cortical-layer name to channel index (see *Cortical-layer index convention*
        in the module docstring). Raises `KeyError` for any other name."""
        return {"deep": 0, "middle": 1, "superficial": 2}[layer]

    @staticmethod
    def _gather_z_hat_at(
        z_hat: torch.Tensor, idx: CollocationBatch, *, signal: HeinzleSignal | int
    ) -> torch.Tensor:
        """Gather one signal's values at `idx`'s collocation points, across every layer.

        Args:
            z_hat: [B, 7, L, T, H, W].
            idx: Shared across every layer -- `idx.h`/`idx.w` broadcast from
                [B, 1, n_times, n_space] against the L axis below.
            signal: Which of the 7 Heinzle channels to gather.

        Returns:
            [B, L, n_times, n_space].
        """
        s = torch.tensor(CollocationMixin._signal_index(signal), device=z_hat.device)
        B, _, L = z_hat.shape[:3]
        b_idx = torch.arange(B, device=z_hat.device)[:, None, None, None]
        s_idx = s[None, None, None, None]
        l_idx = torch.arange(L, device=z_hat.device)[None, :, None, None]
        return z_hat[b_idx, s_idx, l_idx, idx.t, idx.h, idx.w]

    @staticmethod
    def _gather_neural_at(neural: torch.Tensor, idx: CollocationBatch) -> torch.Tensor:
        """Gather ground-truth neural activity at `idx`'s collocation points.

        Args:
            neural: [B, L, T, H, W].
            idx: Shared across every layer, as in `_gather_z_hat_at`.

        Returns:
            [B, L, n_times, n_space].
        """
        B, L = neural.shape[:2]
        b_idx = torch.arange(B, device=neural.device)[:, None, None, None]
        l_idx = torch.arange(L, device=neural.device)[None, :, None, None]
        return neural[b_idx, l_idx, idx.t, idx.h, idx.w]

    @staticmethod
    def _gather_bold_at(bold: torch.Tensor, idx: CollocationBatch) -> torch.Tensor:
        """Gather BOLD values at `idx`'s collocation points, across every layer.

        Args:
            bold: [B, L, T, H, W].
            idx: Shared across every layer, as in `_gather_z_hat_at`.

        Returns:
            [B, L, n_times, n_space].
        """
        B, L = bold.shape[:2]
        b_idx = torch.arange(B, device=bold.device)[:, None, None, None]
        l_idx = torch.arange(L, device=bold.device)[None, :, None, None]
        return bold[b_idx, l_idx, idx.t, idx.h, idx.w]

    @staticmethod
    def _gather_grad_at(
        dz_hat_dt: torch.Tensor, layer: int, idx: CollocationBatch, *, signal: HeinzleSignal | int
    ) -> torch.Tensor:
        """Like `_gather_z_hat_at`, but for one signal's time-derivative at a single
        fixed layer, using that layer's own `idx` (as produced by
        `_sample_collocation_indices_per_layer`) instead of a shared one.

        Args:
            dz_hat_dt: [B, 7, L, T, H, W].
            layer: Fixed layer channel index this `idx` belongs to.
            idx: This layer's own collocation batch (`idx.h`/`idx.w` are
                [B, 1, n_times, n_space]; squeezed here before indexing).
            signal: Which of the 7 Heinzle channels to gather.

        Returns:
            [B, n_times, n_space].
        """
        s = torch.tensor(CollocationMixin._signal_index(signal), device=dz_hat_dt.device)
        B = dz_hat_dt.shape[0]
        b_idx = torch.arange(B, device=dz_hat_dt.device)[:, None, None]
        s_idx = s[None, None, None]
        l_idx = torch.tensor(layer, device=dz_hat_dt.device)[None, None, None]
        t = idx.t.squeeze(1)
        h = idx.h.squeeze(1)
        w = idx.w.squeeze(1)
        return dz_hat_dt[b_idx, s_idx, l_idx, t, h, w]

    @staticmethod
    def _gather_z_hat_at_layer(
        z_hat: torch.Tensor, idx: CollocationBatch, layer: int, *, signal: HeinzleSignal | int
    ) -> torch.Tensor:
        """Like _gather_z_hat_at, but for a single fixed layer using that layer's own
        idx (as produced by _sample_collocation_indices_per_layer) instead of
        broadcasting one shared idx across every layer."""
        s = torch.tensor(CollocationMixin._signal_index(signal), device=z_hat.device)
        B = z_hat.shape[0]
        b_idx = torch.arange(B, device=z_hat.device)[:, None, None]
        s_idx = s[None, None, None]
        l_idx = torch.tensor(layer, device=z_hat.device)[None, None, None]
        t = idx.t.squeeze(1)
        h = idx.h.squeeze(1)
        w = idx.w.squeeze(1)
        return z_hat[b_idx, s_idx, l_idx, t, h, w]

    @staticmethod
    def _gather_bold_at_layer(
        bold: torch.Tensor, idx: CollocationBatch, layer: int
    ) -> torch.Tensor:
        """Like _gather_bold_at, but for a single fixed layer using that layer's own idx."""
        B = bold.shape[0]
        b_idx = torch.arange(B, device=bold.device)[:, None, None]
        l_idx = torch.tensor(layer, device=bold.device)[None, None, None]
        t = idx.t.squeeze(1)
        h = idx.h.squeeze(1)
        w = idx.w.squeeze(1)
        return bold[b_idx, l_idx, t, h, w]

    @staticmethod
    def _full_grid_collocation_batch(
        T: int, H: int, W: int, device: torch.device
    ) -> CollocationBatch:
        """Every `(t, h, w)` in the volume, exactly once -- the `full_grid=True`
        path for both samplers below.

        Slots into the same `[1, 1, n_times, n_space]`-shaped convention as a
        sampled `CollocationBatch`, with `n_times=T`, `n_space=H*W`, so every
        existing `_gather_*` method works unchanged. `t` enumerates true
        chronological order 0..T-1 (broadcast across the H*W axis), so
        `burn_in` slicing downstream still means "skip the first `burn_in`
        real timesteps."
        """
        t = torch.arange(T, device=device).view(1, 1, T, 1).expand(1, 1, T, H * W)
        hw = torch.arange(H * W, device=device)
        h = (hw // W).view(1, 1, 1, H * W).expand(1, 1, T, H * W)
        w = (hw % W).view(1, 1, 1, H * W).expand(1, 1, T, H * W)
        return CollocationBatch(t=t, h=h, w=w)

    @staticmethod
    def _sample_collocation_indices(
        *,
        T: int,
        H: int,
        W: int,
        n_times: int,
        n_space: int,
        device: torch.device,
        source_position: torch.Tensor,
        num_sources: torch.Tensor | None = None,
        dense_spatial_radius: int = 5,
        dense_spatial_frac: float = 0.8,
        dense_time_frac: float = 0.8,
        dense_time_lo: float = 0.05,
        dense_time_hi: float = 0.55,
        uniform_time_lo: float = 0.05,
        full_grid: bool = False,
    ) -> CollocationBatch:
        """Draw one shared (t, h, w) collocation set, reused across every layer.

        Mixes dense (near-source, early/mid-timeline) and uniform draws per the
        *Dense/uniform sampling vocabulary* in the module docstring. If
        `source_position` is None, spatial sampling is uniform-only (no dense
        share) regardless of `dense_spatial_frac`.

        Args:
            T, H, W: Full grid extent to sample within.
            n_times, n_space: Number of collocation timesteps / spatial points
                to draw (independently; the returned batch has n_times x
                n_space (t, h, w) points). Ignored if `full_grid`.
            source_position: [B, S, 2] (h, w) per source, or None for
                source-agnostic (uniform-only) sampling. Ignored if `full_grid`.
            num_sources: [B] valid-source count per sample. Required
                whenever `source_position` is given and the dense share is
                non-empty. Ignored if `full_grid`.
            full_grid: If True, return `_full_grid_collocation_batch(T, H, W,
                device)` directly -- every other argument above is ignored.

        Raises:
            ValueError: If `source_position` is given, the dense spatial share
                is non-empty, and `num_sources` is None. Never raised if
                `full_grid`.

        Returns:
            CollocationBatch with `t`: [1, 1, n_times, n_space], `h`/`w`:
            [1, 1, n_times, n_space] if source-agnostic or [B, 1, n_times,
            n_space] once source-biased (per-sample source positions differ);
            `n_times=T`, `n_space=H*W` if `full_grid`.
        """
        if full_grid:
            return CollocationMixin._full_grid_collocation_batch(T, H, W, device)

        n_dense_t = int(n_times * dense_time_frac)
        n_uniform_t = n_times - n_dense_t

        t_lo_dense = int(T * dense_time_lo)
        t_hi_dense = max(t_lo_dense + 1, int(T * dense_time_hi))
        t_lo_uniform = int(T * uniform_time_lo)

        t_dense = torch.randint(t_lo_dense, t_hi_dense, (n_dense_t, n_space), device=device)
        t_uniform = torch.randint(t_lo_uniform, T, (n_uniform_t, n_space), device=device)
        t = torch.cat([t_dense, t_uniform], dim=0).unsqueeze(0).unsqueeze(0)

        n_dense_s = int(n_space * dense_spatial_frac) if source_position is not None else 0
        n_uniform_s = n_space - n_dense_s

        if n_dense_s > 0:
            if num_sources is None:
                raise ValueError("num_sources is required when source_position is provided")
            B = source_position.shape[0]
            k = num_sources.clamp(min=1)  # [B]

            # Round-robin each dense draw across the sample's active sources so every
            # source gets (near-)equal collocation coverage, rather than an expected
            # share under random per-point source choice.
            n_dense_total = n_times * n_dense_s
            draw_idx = torch.arange(n_dense_total, device=device)
            src_choice = (draw_idx[None, :] % k[:, None]).view(B, n_times, n_dense_s)

            b_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, n_times, n_dense_s)
            src_h = source_position[b_idx, src_choice, 0].long()  # [B, n_times, n_dense_s]
            src_w = source_position[b_idx, src_choice, 1].long()

            off_h = torch.randint(
                -dense_spatial_radius,
                dense_spatial_radius + 1,
                (B, n_times, n_dense_s),
                device=device,
            )
            off_w = torch.randint(
                -dense_spatial_radius,
                dense_spatial_radius + 1,
                (B, n_times, n_dense_s),
                device=device,
            )

            h_dense = (src_h + off_h).clamp(0, H - 1)
            w_dense = (src_w + off_w).clamp(0, W - 1)

            h_uniform = torch.randint(0, H, (B, n_times, n_uniform_s), device=device)
            w_uniform = torch.randint(0, W, (B, n_times, n_uniform_s), device=device)

            h = torch.cat([h_dense, h_uniform], dim=2).unsqueeze(1)
            w = torch.cat([w_dense, w_uniform], dim=2).unsqueeze(1)
        else:
            h = torch.randint(0, H, (1, 1, n_times, n_space), device=device)
            w = torch.randint(0, W, (1, 1, n_times, n_space), device=device)

        return CollocationBatch(t=t, h=h, w=w)

    @staticmethod
    def _sample_collocation_indices_per_layer(
        *,
        T: int,
        H: int,
        W: int,
        L: int,
        n_times: int,
        n_space: int,
        device: torch.device,
        source_position: torch.Tensor,  # [B, S, 2]
        source_layer: torch.Tensor,  # [B, S]
        num_sources: torch.Tensor,  # [B]
        dense_spatial_radius: int = 5,
        dense_spatial_frac: float = 0.8,
        dense_time_frac: float = 0.8,
        dense_time_lo: float = 0.05,
        dense_time_hi: float = 0.55,
        uniform_time_lo: float = 0.05,
        full_grid: bool = False,
    ) -> list[CollocationBatch]:
        """Like _sample_collocation_indices, but draws one independent (t, h, w) set
        per layer instead of sharing a single set across every layer.

        _sample_collocation_indices's dense branch round-robins across every source in
        `source_position` regardless of which layer it's in, then that one shared index
        set gets applied to every layer -- so a layer's own sources share their dense
        collocation budget with every other layer's sources too. Here, each layer's
        dense draws round-robin only over *that layer's own* sources (via
        source_layer). A sample where this layer has zero sources falls back to a
        uniform draw for its whole dense share, instead of clustering around some
        other layer's source location.

        If `full_grid`, every other argument except `L`/`T`/`H`/`W`/`device` is
        ignored and every layer gets the same `_full_grid_collocation_batch`
        (there is no "this layer's own sources" concept once every point is used).
        """
        if full_grid:
            batch = CollocationMixin._full_grid_collocation_batch(T, H, W, device)
            return [batch] * L

        B, S = source_layer.shape
        valid = torch.arange(S, device=device)[None, :] < num_sources[:, None]  # [B, S]

        return [
            CollocationMixin._sample_collocation_indices_one_layer(
                layer=layer,
                T=T,
                H=H,
                W=W,
                n_times=n_times,
                n_space=n_space,
                device=device,
                source_position=source_position,
                source_layer=source_layer,
                valid=valid,
                dense_spatial_radius=dense_spatial_radius,
                dense_spatial_frac=dense_spatial_frac,
                dense_time_frac=dense_time_frac,
                dense_time_lo=dense_time_lo,
                dense_time_hi=dense_time_hi,
                uniform_time_lo=uniform_time_lo,
            )
            for layer in range(L)
        ]

    @staticmethod
    def _sample_collocation_indices_one_layer(
        *,
        layer: int,
        T: int,
        H: int,
        W: int,
        n_times: int,
        n_space: int,
        device: torch.device,
        source_position: torch.Tensor,  # [B, S, 2]
        source_layer: torch.Tensor,  # [B, S]
        valid: torch.Tensor,  # [B, S] bool -- s < num_sources[b]
        dense_spatial_radius: int,
        dense_spatial_frac: float,
        dense_time_frac: float,
        dense_time_lo: float,
        dense_time_hi: float,
        uniform_time_lo: float,
    ) -> CollocationBatch:
        """Draw one layer's (t, h, w) collocation set for `_sample_collocation_indices_per_layer`.

        Same dense/uniform mixing as `_sample_collocation_indices`, but the dense
        spatial share round-robins only over sources where `source_layer == layer`
        (via `valid`, precomputed once by the caller). A sample with zero sources
        in this layer gets a uniform fallback draw for its entire dense share
        instead of clustering around another layer's source.

        Args:
            layer: This call's fixed layer index.
            valid: [B, S] bool, True where slot s is a real (non-padding) source
                for that sample, precomputed by the caller from `num_sources`.

        Returns:
            CollocationBatch with `t`: [1, 1, n_times, n_space], `h`/`w`: [B, 1,
            n_times, n_space].
        """
        n_dense_t = int(n_times * dense_time_frac)
        n_uniform_t = n_times - n_dense_t

        t_lo_dense = int(T * dense_time_lo)
        t_hi_dense = max(t_lo_dense + 1, int(T * dense_time_hi))
        t_lo_uniform = int(T * uniform_time_lo)

        t_dense = torch.randint(t_lo_dense, t_hi_dense, (n_dense_t, n_space), device=device)
        t_uniform = torch.randint(t_lo_uniform, T, (n_uniform_t, n_space), device=device)
        t = torch.cat([t_dense, t_uniform], dim=0).unsqueeze(0).unsqueeze(0)

        B = source_position.shape[0]
        n_dense_s = int(n_space * dense_spatial_frac)
        n_uniform_s = n_space - n_dense_s

        if n_dense_s > 0:
            belongs = valid & (source_layer == layer)  # [B, S] -- this layer's own sources
            k_layer = belongs.sum(dim=1)  # [B]
            has_layer_source = k_layer > 0  # [B]
            k_safe = k_layer.clamp(min=1)

            # Sort so this layer's own source slots come first; round-robin only over
            # those (mirrors _sample_collocation_indices's round-robin, but scoped to
            # one layer's sources instead of every source pooled together).
            order = torch.argsort((~belongs).float(), dim=1, stable=True)  # [B, S]
            n_dense_total = n_times * n_dense_s
            draw_idx = torch.arange(n_dense_total, device=device)
            slot = (draw_idx[None, :] % k_safe[:, None]).view(B, n_times, n_dense_s)
            b_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, n_times, n_dense_s)
            src_choice = order[b_idx, slot]  # [B, n_times, n_dense_s] -- index into S

            src_h = source_position[b_idx, src_choice, 0].long()
            src_w = source_position[b_idx, src_choice, 1].long()
            off_h = torch.randint(
                -dense_spatial_radius,
                dense_spatial_radius + 1,
                (B, n_times, n_dense_s),
                device=device,
            )
            off_w = torch.randint(
                -dense_spatial_radius,
                dense_spatial_radius + 1,
                (B, n_times, n_dense_s),
                device=device,
            )
            h_dense = (src_h + off_h).clamp(0, H - 1)
            w_dense = (src_w + off_w).clamp(0, W - 1)

            # Samples where this layer has no source of its own: fall back to a
            # uniform draw instead of clustering "densely" around a different layer's
            # source location.
            h_fallback = torch.randint(0, H, (B, n_times, n_dense_s), device=device)
            w_fallback = torch.randint(0, W, (B, n_times, n_dense_s), device=device)
            keep = has_layer_source.view(B, 1, 1)
            h_dense = torch.where(keep, h_dense, h_fallback)
            w_dense = torch.where(keep, w_dense, w_fallback)
        else:
            h_dense = torch.empty(B, n_times, 0, dtype=torch.long, device=device)
            w_dense = torch.empty(B, n_times, 0, dtype=torch.long, device=device)

        h_uniform = torch.randint(0, H, (B, n_times, n_uniform_s), device=device)
        w_uniform = torch.randint(0, W, (B, n_times, n_uniform_s), device=device)

        h = torch.cat([h_dense, h_uniform], dim=2).unsqueeze(1)
        w = torch.cat([w_dense, w_uniform], dim=2).unsqueeze(1)

        return CollocationBatch(t=t, h=h, w=w)
