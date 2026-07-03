"""Collocation-point sampling and index-based gathering into [B, ..., L, T, H, W] tensors."""

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
        return torch.linspace(0.0, 1.0, T, device=device, dtype=dtype).view(1, T).expand(B, T)

    @staticmethod
    def _signal_index(signal: HeinzleSignal | int) -> int:
        mapping = {"x": 0, "s": 1, "f": 2, "v": 3, "q": 4, "vstar": 5, "qstar": 6}
        if isinstance(signal, int):
            if 0 <= signal < 7:
                return signal
            raise IndexError(f"signal index must be in [0,6], got {signal}")
        return mapping[signal]

    @staticmethod
    def _layer_index(layer: str) -> int:
        return {"deep": 0, "middle": 1, "superficial": 2}[layer]

    @staticmethod
    def _gather_z_hat_at(
        z_hat: torch.Tensor, idx: CollocationBatch, *, signal: HeinzleSignal | int
    ) -> torch.Tensor:
        s = torch.tensor(CollocationMixin._signal_index(signal), device=z_hat.device)
        B, _, L = z_hat.shape[:3]
        b_idx = torch.arange(B, device=z_hat.device)[:, None, None, None]
        s_idx = s[None, None, None, None]
        l_idx = torch.arange(L, device=z_hat.device)[None, :, None, None]
        return z_hat[b_idx, s_idx, l_idx, idx.t, idx.h, idx.w]

    @staticmethod
    def _gather_neural_at(neural: torch.Tensor, idx: CollocationBatch) -> torch.Tensor:
        B, L = neural.shape[:2]
        b_idx = torch.arange(B, device=neural.device)[:, None, None, None]
        l_idx = torch.arange(L, device=neural.device)[None, :, None, None]
        return neural[b_idx, l_idx, idx.t, idx.h, idx.w]

    @staticmethod
    def _gather_bold_at(bold: torch.Tensor, idx: CollocationBatch) -> torch.Tensor:
        B, L = bold.shape[:2]
        b_idx = torch.arange(B, device=bold.device)[:, None, None, None]
        l_idx = torch.arange(L, device=bold.device)[None, :, None, None]
        return bold[b_idx, l_idx, idx.t, idx.h, idx.w]

    @staticmethod
    def _gather_grad_at(
        dz_hat_dt: torch.Tensor, layer: int, idx: CollocationBatch, *, signal: HeinzleSignal | int
    ) -> torch.Tensor:
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
    ) -> CollocationBatch:
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
