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


def test_gather_z_hat_at_layer_matches_manual_indexing():
    B, S, L, T, H, W = 2, 7, 3, 6, 4, 4
    z_hat = torch.randn(B, S, L, T, H, W)
    n_t, n_s = 4, 2
    idx = CollocationBatch(
        t=torch.randint(0, T, (1, 1, n_t, n_s)),
        h=torch.randint(0, H, (B, 1, n_t, n_s)),
        w=torch.randint(0, W, (B, 1, n_t, n_s)),
    )
    layer = 2
    gathered = CollocationMixin._gather_z_hat_at_layer(z_hat, idx, layer, signal="f")
    assert gathered.shape == (B, n_t, n_s)
    f_idx = CollocationMixin._signal_index("f")
    expected = z_hat[1, f_idx, layer, idx.t[0, 0, -1, -1], idx.h[1, 0, -1, -1], idx.w[1, 0, -1, -1]]
    assert torch.equal(gathered[1, -1, -1], expected)


def test_gather_bold_at_layer_matches_manual_indexing():
    B, L, T, H, W = 2, 3, 6, 4, 4
    bold = torch.randn(B, L, T, H, W)
    n_t, n_s = 4, 2
    idx = CollocationBatch(
        t=torch.randint(0, T, (1, 1, n_t, n_s)),
        h=torch.randint(0, H, (B, 1, n_t, n_s)),
        w=torch.randint(0, W, (B, 1, n_t, n_s)),
    )
    layer = 1
    gathered = CollocationMixin._gather_bold_at_layer(bold, idx, layer)
    assert gathered.shape == (B, n_t, n_s)
    expected = bold[0, layer, idx.t[0, 0, 0, 0], idx.h[0, 0, 0, 0], idx.w[0, 0, 0, 0]]
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


# -----------------------------
# _sample_collocation_indices: full_grid
# -----------------------------


def test_sample_collocation_indices_full_grid_covers_every_point_exactly_once():
    T, H, W = 4, 3, 2
    idx = CollocationMixin._sample_collocation_indices(
        T=T,
        H=H,
        W=W,
        n_times=999,  # ignored
        n_space=999,  # ignored
        device=torch.device("cpu"),
        source_position=None,
        dense_spatial_frac=0.5,  # ignored
        full_grid=True,
    )
    assert idx.t.shape == (1, 1, T, H * W)
    assert idx.h.shape == (1, 1, T, H * W)
    assert idx.w.shape == (1, 1, T, H * W)

    triples = {
        (int(idx.t[0, 0, i, j]), int(idx.h[0, 0, i, j]), int(idx.w[0, 0, i, j]))
        for i in range(T)
        for j in range(H * W)
    }
    expected = {(t, h, w) for t in range(T) for h in range(H) for w in range(W)}
    assert triples == expected


def test_sample_collocation_indices_full_grid_ignores_source_requirements():
    """full_grid=True bypasses source_position/num_sources validation entirely --
    this would normally raise (see test_..._requires_num_sources_with_source_position)."""
    source_position = torch.zeros(2, 1, 2, dtype=torch.long)
    idx = CollocationMixin._sample_collocation_indices(
        T=3,
        H=2,
        W=2,
        n_times=1,
        n_space=1,
        device=torch.device("cpu"),
        source_position=source_position,
        num_sources=None,
        full_grid=True,
    )
    assert idx.t.shape == (1, 1, 3, 4)


# -----------------------------
# _sample_collocation_indices_per_layer
# -----------------------------


def test_sample_collocation_indices_per_layer_returns_one_batch_per_layer_with_right_shapes():
    B, S, L = 2, 2, 3
    H, W = 20, 20
    source_position = torch.randint(0, H, (B, S, 2))
    source_layer = torch.randint(0, L, (B, S))
    num_sources = torch.full((B,), S, dtype=torch.long)

    idx_list = CollocationMixin._sample_collocation_indices_per_layer(
        T=10,
        H=H,
        W=W,
        L=L,
        n_times=5,
        n_space=6,
        device=torch.device("cpu"),
        source_position=source_position,
        source_layer=source_layer,
        num_sources=num_sources,
        dense_spatial_frac=0.5,
        dense_spatial_radius=2,
    )
    assert len(idx_list) == L
    for idx in idx_list:
        assert idx.h.shape == (B, 1, 5, 6)
        assert idx.w.shape == (B, 1, 5, 6)
        assert idx.h.min() >= 0 and idx.h.max() < H
        assert idx.w.min() >= 0 and idx.w.max() < W


def test_sample_collocation_indices_per_layer_dense_draws_use_only_this_layers_sources():
    """A layer's dense collocation points must cluster around its own source(s) only --
    not another layer's source, even though _sample_collocation_indices' pooled
    round-robin would mix them."""
    H, W = 100, 100
    radius = 2
    # Layer 0's source and layer 1's source sit far apart so we can tell which one any
    # given dense draw clustered around.
    source_position = torch.tensor([[[10, 10], [80, 80]]], dtype=torch.long)  # [B, S=2, 2]
    source_layer = torch.tensor([[0, 1]])  # source 0 -> layer 0, source 1 -> layer 1
    num_sources = torch.tensor([2])

    idx_list = CollocationMixin._sample_collocation_indices_per_layer(
        T=10,
        H=H,
        W=W,
        L=2,
        n_times=6,
        n_space=6,
        device=torch.device("cpu"),
        source_position=source_position,
        source_layer=source_layer,
        num_sources=num_sources,
        dense_spatial_frac=1.0,  # all points dense
        dense_spatial_radius=radius,
    )
    h0, w0 = idx_list[0].h[0, 0].reshape(-1), idx_list[0].w[0, 0].reshape(-1)
    h1, w1 = idx_list[1].h[0, 0].reshape(-1), idx_list[1].w[0, 0].reshape(-1)

    assert torch.all((h0 - 10).abs() <= radius) and torch.all((w0 - 10).abs() <= radius)
    assert torch.all((h1 - 80).abs() <= radius) and torch.all((w1 - 80).abs() <= radius)


def test_sample_collocation_indices_per_layer_falls_back_to_uniform_without_own_source():
    """A sample where this layer has zero sources of its own must not cluster around a
    different layer's source location -- it should fall back to a uniform draw."""
    H, W = 100, 100
    radius = 2
    # Only one source, belonging to layer 0. Layer 1 has none.
    source_position = torch.tensor([[[10, 10]]], dtype=torch.long)  # [B, S=1, 2]
    source_layer = torch.tensor([[0]])
    num_sources = torch.tensor([1])

    idx_list = CollocationMixin._sample_collocation_indices_per_layer(
        T=10,
        H=H,
        W=W,
        L=2,
        n_times=6,
        n_space=20,
        device=torch.device("cpu"),
        source_position=source_position,
        source_layer=source_layer,
        num_sources=num_sources,
        dense_spatial_frac=1.0,
        dense_spatial_radius=radius,
    )
    h1 = idx_list[1].h[0, 0].reshape(-1)
    w1 = idx_list[1].w[0, 0].reshape(-1)
    near_layer0_source = (h1 - 10).abs() <= radius
    # With a uniform fallback over a 100x100 grid and 120 draws, landing within a
    # radius-2 box (25 cells) of (10, 10) on every single draw is not a real fallback.
    assert not torch.all(near_layer0_source)
    assert w1.max() > 10 + radius or h1.max() > 10 + radius


# -----------------------------
# _sample_collocation_indices_per_layer: full_grid
# -----------------------------


def test_sample_collocation_indices_per_layer_full_grid_returns_same_batch_for_every_layer():
    T, H, W, L = 3, 2, 2, 4
    source_position = torch.zeros(2, 1, 2, dtype=torch.long)
    source_layer = torch.zeros(2, 1, dtype=torch.long)
    num_sources = torch.ones(2, dtype=torch.long)

    idx_list = CollocationMixin._sample_collocation_indices_per_layer(
        T=T,
        H=H,
        W=W,
        L=L,
        n_times=1,  # ignored
        n_space=1,  # ignored
        device=torch.device("cpu"),
        source_position=source_position,
        source_layer=source_layer,
        num_sources=num_sources,
        full_grid=True,
    )
    assert len(idx_list) == L
    assert all(batch is idx_list[0] for batch in idx_list)  # one shared batch, reused

    idx = idx_list[0]
    triples = {
        (int(idx.t[0, 0, i, j]), int(idx.h[0, 0, i, j]), int(idx.w[0, 0, i, j]))
        for i in range(T)
        for j in range(H * W)
    }
    expected = {(t, h, w) for t in range(T) for h in range(H) for w in range(W)}
    assert triples == expected
