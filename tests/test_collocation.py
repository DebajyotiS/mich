"""Direct unit tests for CollocationMixin (src/mich/models/collocation.py)."""

from __future__ import annotations

import pytest
import torch
from mich.models.collocation import CollocationBatch, CollocationMixin


class _C(CollocationMixin):
    pass


# -----------------------------
# _make_time_grid
# -----------------------------


def test_make_time_grid_shape_and_range():
    t = _C._make_time_grid(B=3, T=5, device=torch.device("cpu"), dtype=torch.float32)
    assert t.shape == (3, 5)
    assert torch.allclose(t[0], torch.linspace(0.0, 1.0, 5))
    assert torch.equal(t[0], t[1])  # expanded, identical across batch
    assert t.dtype == torch.float32


# -----------------------------
# _signal_index / _layer_index
# -----------------------------


@pytest.mark.parametrize(
    "name,expected",
    [("x", 0), ("s", 1), ("f", 2), ("v", 3), ("q", 4), ("vstar", 5), ("qstar", 6)],
)
def test_signal_index_all_names(name, expected):
    assert _C._signal_index(name) == expected


@pytest.mark.parametrize("i", [0, 3, 6])
def test_signal_index_int_passthrough(i):
    assert _C._signal_index(i) == i


@pytest.mark.parametrize("i", [-1, 7, 100])
def test_signal_index_int_out_of_range_raises(i):
    with pytest.raises(IndexError):
        _C._signal_index(i)


def test_signal_index_unknown_string_raises():
    with pytest.raises(KeyError):
        _C._signal_index("bogus")


@pytest.mark.parametrize("name,expected", [("deep", 0), ("middle", 1), ("superficial", 2)])
def test_layer_index_all_names(name, expected):
    assert _C._layer_index(name) == expected


def test_layer_index_unknown_raises():
    with pytest.raises(KeyError):
        _C._layer_index("bogus")


# -----------------------------
# _gather_* correctness
# -----------------------------


def test_gather_z_hat_at_matches_manual_indexing():
    B, S, L, T, H, W = 2, 7, 3, 6, 4, 4
    z_hat = torch.randn(B, S, L, T, H, W)

    n_t, n_s = 5, 2
    t_idx = torch.randint(0, T, (1, 1, n_t, n_s))
    h_idx = torch.randint(0, H, (B, 1, n_t, n_s))
    w_idx = torch.randint(0, W, (B, 1, n_t, n_s))
    idx = CollocationBatch(t=t_idx, h=h_idx, w=w_idx)

    gathered = CollocationMixin._gather_z_hat_at(z_hat, idx, signal="v")
    assert gathered.shape == (B, L, n_t, n_s)

    v_idx = CollocationMixin._signal_index("v")
    for b in range(B):
        for layer in range(L):
            for it in range(n_t):
                for isp in range(n_s):
                    expected = z_hat[
                        b,
                        v_idx,
                        layer,
                        t_idx[0, 0, it, isp],
                        h_idx[b, 0, it, isp],
                        w_idx[b, 0, it, isp],
                    ]
                    assert torch.equal(gathered[b, layer, it, isp], expected)


def test_gather_neural_at_matches_manual_indexing():
    B, L, T, H, W = 2, 3, 6, 4, 4
    neural = torch.randn(B, L, T, H, W)
    n_t, n_s = 4, 2
    idx = CollocationBatch(
        t=torch.randint(0, T, (1, 1, n_t, n_s)),
        h=torch.randint(0, H, (B, 1, n_t, n_s)),
        w=torch.randint(0, W, (B, 1, n_t, n_s)),
    )
    gathered = CollocationMixin._gather_neural_at(neural, idx)
    assert gathered.shape == (B, L, n_t, n_s)
    expected = neural[0, 0, idx.t[0, 0, 0, 0], idx.h[0, 0, 0, 0], idx.w[0, 0, 0, 0]]
    assert torch.equal(gathered[0, 0, 0, 0], expected)


def test_gather_bold_at_matches_manual_indexing():
    B, L, T, H, W = 2, 3, 6, 4, 4
    bold = torch.randn(B, L, T, H, W)
    n_t, n_s = 4, 2
    idx = CollocationBatch(
        t=torch.randint(0, T, (1, 1, n_t, n_s)),
        h=torch.randint(0, H, (B, 1, n_t, n_s)),
        w=torch.randint(0, W, (B, 1, n_t, n_s)),
    )
    gathered = CollocationMixin._gather_bold_at(bold, idx)
    assert gathered.shape == (B, L, n_t, n_s)
    expected = bold[1, 2, idx.t[0, 0, -1, -1], idx.h[1, 0, -1, -1], idx.w[1, 0, -1, -1]]
    assert torch.equal(gathered[1, 2, -1, -1], expected)


def test_gather_grad_at_matches_manual_indexing_for_fixed_layer():
    B, S, L, T, H, W = 2, 7, 3, 6, 4, 4
    dz_hat_dt = torch.randn(B, S, L, T, H, W)
    n_t, n_s = 4, 2
    idx = CollocationBatch(
        t=torch.randint(0, T, (1, 1, n_t, n_s)),
        h=torch.randint(0, H, (B, 1, n_t, n_s)),
        w=torch.randint(0, W, (B, 1, n_t, n_s)),
    )
    layer = 1
    gathered = CollocationMixin._gather_grad_at(dz_hat_dt, layer, idx, signal="q")
    assert gathered.shape == (B, n_t, n_s)
    q_idx = CollocationMixin._signal_index("q")
    expected = dz_hat_dt[0, q_idx, layer, idx.t[0, 0, 0, 0], idx.h[0, 0, 0, 0], idx.w[0, 0, 0, 0]]
    assert torch.equal(gathered[0, 0, 0], expected)


# -----------------------------
# _sample_collocation_indices
# -----------------------------


def test_sample_collocation_indices_shapes_without_source_position():
    idx = CollocationMixin._sample_collocation_indices(
        T=20,
        H=8,
        W=8,
        n_times=6,
        n_space=4,
        device=torch.device("cpu"),
        source_position=None,
    )
    assert idx.t.shape == (1, 1, 6, 4)
    assert idx.h.shape == (1, 1, 6, 4)
    assert idx.w.shape == (1, 1, 6, 4)
    assert idx.h.min() >= 0 and idx.h.max() < 8
    assert idx.w.min() >= 0 and idx.w.max() < 8


def test_sample_collocation_indices_requires_num_sources_with_source_position():
    source_position = torch.zeros(2, 1, 2, dtype=torch.long)
    with pytest.raises(ValueError, match="num_sources is required"):
        CollocationMixin._sample_collocation_indices(
            T=20,
            H=8,
            W=8,
            n_times=6,
            n_space=4,
            device=torch.device("cpu"),
            source_position=source_position,
            num_sources=None,
        )


def test_sample_collocation_indices_time_ranges_dense_vs_uniform():
    T = 100
    n_times, n_space = 10, 1
    dense_time_frac = 0.7
    dense_time_lo, dense_time_hi, uniform_time_lo = 0.1, 0.5, 0.2

    idx = CollocationMixin._sample_collocation_indices(
        T=T,
        H=8,
        W=8,
        n_times=n_times,
        n_space=n_space,
        device=torch.device("cpu"),
        source_position=None,
        dense_time_frac=dense_time_frac,
        dense_time_lo=dense_time_lo,
        dense_time_hi=dense_time_hi,
        uniform_time_lo=uniform_time_lo,
    )
    n_dense_t = int(n_times * dense_time_frac)
    t = idx.t[0, 0]  # [n_times, n_space]
    dense_part = t[:n_dense_t]
    uniform_part = t[n_dense_t:]

    t_lo_dense = int(T * dense_time_lo)
    t_hi_dense = max(t_lo_dense + 1, int(T * dense_time_hi))
    t_lo_uniform = int(T * uniform_time_lo)

    assert dense_part.min() >= t_lo_dense and dense_part.max() < t_hi_dense
    assert uniform_part.min() >= t_lo_uniform and uniform_part.max() < T


def test_sample_collocation_indices_dense_spatial_clusters_around_sources():
    B = 2
    H, W = 50, 50
    source_position = torch.tensor([[[25, 30]], [[10, 40]]], dtype=torch.long)
    num_sources = torch.ones(B, dtype=torch.long)
    radius = 3

    idx = CollocationMixin._sample_collocation_indices(
        T=10,
        H=H,
        W=W,
        n_times=5,
        n_space=10,
        device=torch.device("cpu"),
        source_position=source_position,
        num_sources=num_sources,
        dense_spatial_frac=1.0,  # all points dense -> all must cluster around the source
        dense_spatial_radius=radius,
    )
    # idx.h/idx.w: [B, 1, n_times, n_space]
    for b in range(B):
        src_h, src_w = source_position[b, 0, 0].item(), source_position[b, 0, 1].item()
        h_vals = idx.h[b, 0]
        w_vals = idx.w[b, 0]
        assert (h_vals - src_h).abs().max() <= radius
        assert (w_vals - src_w).abs().max() <= radius


def test_sample_collocation_indices_clamps_dense_spatial_near_boundary():
    B = 1
    H, W = 10, 10
    # Source right at the corner -- offsets must clamp into [0, H-1]/[0, W-1], not go negative/overflow.
    source_position = torch.tensor([[[0, 0]]], dtype=torch.long)
    num_sources = torch.ones(B, dtype=torch.long)

    idx = CollocationMixin._sample_collocation_indices(
        T=10,
        H=H,
        W=W,
        n_times=5,
        n_space=8,
        device=torch.device("cpu"),
        source_position=source_position,
        num_sources=num_sources,
        dense_spatial_frac=1.0,
        dense_spatial_radius=5,
    )
    assert idx.h.min() >= 0 and idx.h.max() < H
    assert idx.w.min() >= 0 and idx.w.max() < W


def test_sample_collocation_indices_round_robins_across_multiple_sources():
    H, W = 100, 100
    # Widely separated sources so we can tell which one each dense point clustered around.
    source_position = torch.tensor([[[5, 5], [50, 50], [90, 90]]], dtype=torch.long)
    num_sources = torch.tensor([3])

    idx = CollocationMixin._sample_collocation_indices(
        T=10,
        H=H,
        W=W,
        n_times=6,
        n_space=6,
        device=torch.device("cpu"),
        source_position=source_position,
        num_sources=num_sources,
        dense_spatial_frac=1.0,
        dense_spatial_radius=2,
    )
    h_vals = idx.h[0, 0].reshape(-1)
    w_vals = idx.w[0, 0].reshape(-1)
    near_src0 = ((h_vals - 5).abs() <= 2) & ((w_vals - 5).abs() <= 2)
    near_src1 = ((h_vals - 50).abs() <= 2) & ((w_vals - 50).abs() <= 2)
    near_src2 = ((h_vals - 90).abs() <= 2) & ((w_vals - 90).abs() <= 2)
    # Every point must be near exactly one of the three sources (round-robin coverage).
    assert torch.all(near_src0 | near_src1 | near_src2)
    assert near_src0.any() and near_src1.any() and near_src2.any()


def test_sample_collocation_indices_partial_dense_frac_produces_uniform_tail():
    B, _S = 1, 1
    H, W = 100, 100
    source_position = torch.tensor([[[50, 50]]], dtype=torch.long)
    num_sources = torch.ones(B, dtype=torch.long)

    idx = CollocationMixin._sample_collocation_indices(
        T=10,
        H=H,
        W=W,
        n_times=1,
        n_space=10,
        device=torch.device("cpu"),
        source_position=source_position,
        num_sources=num_sources,
        dense_spatial_frac=0.5,  # 5 dense, 5 uniform
        dense_spatial_radius=2,
    )
    h_vals = idx.h[0, 0, 0]  # [n_space]
    _w_vals = idx.w[0, 0, 0]
    near_source = (h_vals - 50).abs() <= 2
    # Not all points should be forced near the source when dense_spatial_frac < 1.
    assert not torch.all(near_source)
