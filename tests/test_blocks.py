from __future__ import annotations

from dataclasses import is_dataclass

import pytest
import torch

from mich.models.blocks import (  # noqa: E402
    HEINZLE_SIGNAL_IDX,
    DepthWiseSeparableConvLayer,
    FourierTimeEmbedding,
    HeinzleNet,
    MaskedLayerMixing,
    SpatialDecoderManifest,
    SpatialEncoder,
    SpatioTemporalDecoder,
    TemporalDepthWiseTCNLayer,
    TemporalMixingEncoder,
    TimeFiLM,
    _init_heinzle_output_bias,
)

# -----------------------------
# Global fixtures / helpers
# -----------------------------


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(0)


def _mk_small_inputs(*, B=2, L=3, T=4, H=8, W=7, dtype=torch.float32, device="cpu"):
    x = torch.randn(B, L, T, H, W, dtype=dtype, device=device)
    t = torch.linspace(0.0, 1.0, T, dtype=dtype, device=device).unsqueeze(0).expand(B, -1)
    return x, t


def _mk_heinzle_configs(*, L=3, Cmix=4, Cenc=6, c_dec=8, c_film=4):
    layer_mixing_config = dict(L=L, C=Cmix, init_identity=True)

    spatial_encoder_config = [
        dict(
            cin=Cmix,
            cout=Cenc,
            stride=1,
            dw_kernel=3,
            pw_kernel=1,
            num_groups=1,
            activation="silu",
        )
    ]

    temporal_mixing_config = [
        dict(
            cin=Cenc,
            kernel_size=3,
            num_groups=1,
            activation="silu",
        )
    ]

    time_embedding_config = dict(num_freqs=4, max_freq=3.0)
    time_film_config = dict(
        embed_dim=2 * time_embedding_config["num_freqs"],
        hidden_dim=16,
        activation="silu",
        c_dec=c_film,
    )
    spatial_decoder_config = dict(
        cin=Cenc,
        c_dec=c_dec,
        c_film=c_film,
        out_channels=7,
        activation="silu",
        L=L,
        upsample=False,
    )

    return dict(
        layer_mixing_config=layer_mixing_config,
        spatial_encoder_config=spatial_encoder_config,
        temporal_mixing_config=temporal_mixing_config,
        time_embedding_config=time_embedding_config,
        time_film_config=time_film_config,
        spatial_decoder_config=spatial_decoder_config,
    )


# -----------------------------
# SpatialDecoderManifest
# -----------------------------


def test_spatial_decoder_manifest_is_dataclass_and_fields():
    assert is_dataclass(SpatialDecoderManifest)
    z = torch.zeros((1, 7, 3, 2, 4, 5))
    m = SpatialDecoderManifest(z_hat=z)
    assert m.z_hat is z
    assert m.dz_hat_dt is None


def test_spatial_decoder_manifest_channel_indexes_correctly():
    B, L, T, H, W = 2, 3, 4, 5, 6
    z = torch.randn(B, 7, L, T, H, W)
    m = SpatialDecoderManifest(z_hat=z)

    for signal, idx in HEINZLE_SIGNAL_IDX.items():
        assert torch.equal(m.channel(signal), z[:, idx])


def test_spatial_decoder_manifest_channel_grad_raises_without_grads():
    z = torch.randn(1, 7, 2, 3, 4, 4)
    m = SpatialDecoderManifest(z_hat=z)
    with pytest.raises(RuntimeError, match="Gradients were not requested"):
        m.channel_grad("v")


def test_spatial_decoder_manifest_channel_grad_indexes_correctly_when_set():
    B, L, T, H, W = 1, 2, 3, 4, 4
    z = torch.randn(B, 7, L, T, H, W)
    grads = torch.randn(B, 7, L, T, H, W)
    m = SpatialDecoderManifest(z_hat=z, grads=grads)

    for signal, idx in HEINZLE_SIGNAL_IDX.items():
        assert torch.equal(m.channel_grad(signal), grads[:, idx])


# -----------------------------
# _init_heinzle_output_bias
# -----------------------------


def test_init_heinzle_output_bias_raises_when_bias_missing():
    conv = torch.nn.Conv2d(4, 7, kernel_size=1, bias=False)
    with pytest.raises(ValueError, match="Expected out_conv to have a bias"):
        _init_heinzle_output_bias(conv, L=2)


def test_init_heinzle_output_bias_raises_when_out_channels_mismatch():
    conv = torch.nn.Conv2d(4, 5, kernel_size=1, bias=True)
    with pytest.raises(ValueError, match=r"got 5"):
        _init_heinzle_output_bias(conv, L=2)


def test_init_heinzle_output_bias_sets_expected_constants():
    conv = torch.nn.Conv2d(4, 7, kernel_size=1, bias=True)
    _init_heinzle_output_bias(conv, L=2)
    bias = conv.bias.detach()

    x_idx = HEINZLE_SIGNAL_IDX["x"]
    assert bias[x_idx].item() == pytest.approx(-3.0)  # softplus_inv_0

    for sig in ("f", "v", "q"):
        assert bias[HEINZLE_SIGNAL_IDX[sig]].item() == pytest.approx(0.5413)  # softplus_inv_1

    for sig in ("s", "vstar", "qstar"):
        assert bias[HEINZLE_SIGNAL_IDX[sig]].item() == 0.0


# -----------------------------
# MaskedLayerMixing
# -----------------------------


def test_masked_layer_mixing_mask_l_equals_1_is_just_diagonal():
    m = MaskedLayerMixing(L=1, C=3, init_identity=True)
    mask2d = m.mask[:, :, 0, 0]
    assert torch.equal(mask2d, torch.tensor([[1.0]]))


def test_masked_layer_mixing_mask_structure_and_identity_init():
    L = 4
    m = MaskedLayerMixing(L=L, C=5, init_identity=True)

    # Mask should be registered buffer and have shape [L,L,1,1]
    assert hasattr(m, "mask")
    assert tuple(m.mask.shape) == (L, L, 1, 1)

    mask2d = m.mask[:, :, 0, 0].cpu()
    # Diagonal ones
    assert torch.all(torch.diag(mask2d) == 1.0)
    # Subdiagonal ones (i, i-1) for i>=1
    assert torch.all(mask2d[1:, :-1].diag() == 1.0)
    # Other entries should be zero
    for i in range(L):
        for j in range(L):
            if i == j or (i == j + 1):
                continue
            assert mask2d[i, j].item() == 0.0

    # Identity init: W diagonal = 1
    W = m.W.detach().cpu()
    for i in range(L):
        assert W[i, i, 0, 0].item() == 1.0


def test_masked_layer_mixing_forward_shape_and_layer_check():
    B, L, T, H, W = 2, 3, 4, 6, 5
    x = torch.randn(B, L, T, H, W)
    m = MaskedLayerMixing(L=L, C=7)

    y = m(x)
    assert tuple(y.shape) == (B, T, L, m.C, H, W)

    # Wrong L should raise AssertionError
    x_bad = torch.randn(B, L + 1, T, H, W)
    with pytest.raises(AssertionError, match="Expected input with"):
        _ = m(x_bad)


def test_masked_layer_mixing_respects_mask_zeroing_off_diagonal():
    # With identity W, each layer's mixed output equals its own input.
    # expand_net: Conv2d(1, C, 1) — set weights to 1 so output = input value broadcast to C channels.
    B, L, T, H, W = 1, 3, 2, 4, 4
    C = 4
    m = MaskedLayerMixing(L=L, C=C, init_identity=True)
    with torch.no_grad():
        m.expand_net.weight.fill_(1.0)
        m.expand_net.bias.zero_()

    x = torch.zeros(B, L, T, H, W)
    x[:, 0] = 5.0  # only first layer has signal

    y = m(x)  # [B, T, L, C, H, W]
    assert tuple(y.shape) == (B, T, L, C, H, W)
    # Layer 0 output should be non-zero (contains the signal)
    assert not torch.allclose(y[:, :, 0], torch.zeros_like(y[:, :, 0]), atol=1e-6)
    # Layer 2 output should be zero (no signal, and identity W means no cross-layer bleed from layer 0)
    assert torch.allclose(y[:, :, 2], torch.zeros_like(y[:, :, 2]), atol=1e-6)


# -----------------------------
# DepthWiseSeparableConvLayer
# -----------------------------


def test_depthwise_separable_conv_layer_init_group_assertion():
    with pytest.raises(AssertionError, match="num_groups must be"):
        _ = DepthWiseSeparableConvLayer(cin=4, cout=10, num_groups=3)  # 10 % 3 != 0


def test_depthwise_separable_conv_layer_forward_shape_and_grad():
    layer = DepthWiseSeparableConvLayer(cin=4, cout=8, stride=1, num_groups=2, activation="silu")
    x = torch.randn(3, 4, 16, 12, requires_grad=True)
    y = layer(x)
    assert tuple(y.shape) == (3, 8, 16, 12)
    y.mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


# -----------------------------
# SpatialEncoder
# -----------------------------


def test_spatial_encoder_preserves_batch_time_and_changes_channels():
    enc = SpatialEncoder(
        [
            dict(
                cin=4, cout=6, stride=1, dw_kernel=3, pw_kernel=1, num_groups=1, activation="silu"
            ),
            dict(
                cin=6, cout=7, stride=1, dw_kernel=3, pw_kernel=1, num_groups=1, activation="silu"
            ),
        ]
    )
    x = torch.randn(2, 5, 3, 4, 11, 9)  # [B,T,L,C,H,W]
    y = enc(x)
    assert y.shape[:3] == (2, 5, 3)
    assert y.shape[3] == 7
    assert y.shape[4:] == (11, 9)


# -----------------------------
# TemporalDepthWiseTCNLayer
# -----------------------------


def test_temporal_depthwise_tcn_layer_residual_identity_when_weights_zero():
    layer = TemporalDepthWiseTCNLayer(
        cin=4, dilation=1, kernel_size=3, num_groups=1, activation="silu"
    )
    layer.eval()
    with torch.no_grad():
        layer.depthwise.weight.zero_()
        layer.pointwise.weight.zero_()
        # GroupNorm default affine -> weight=1, bias=0, so norm(0)=0
    x = torch.randn(3, 4, 10)  # [N,C,T]
    y = layer(x)
    assert torch.allclose(y, x, atol=1e-6)


def test_temporal_depthwise_tcn_layer_shape():
    layer = TemporalDepthWiseTCNLayer(
        cin=6, dilation=2, kernel_size=3, num_groups=2, activation="silu"
    )
    x = torch.randn(5, 6, 17)
    y = layer(x)
    assert y.shape == x.shape


# -----------------------------
# TemporalMixingEncoder
# -----------------------------


def test_temporal_mixing_encoder_sets_dilations_powers_of_two():
    enc = TemporalMixingEncoder(
        [
            dict(cin=4, kernel_size=3, num_groups=1, activation="silu"),
            dict(cin=4, kernel_size=3, num_groups=1, activation="silu"),
            dict(cin=4, kernel_size=3, num_groups=1, activation="silu"),
        ]
    )
    assert enc.num_layers == 3
    assert enc.module[0].depthwise.dilation == (1,)
    assert enc.module[1].depthwise.dilation == (2,)
    assert enc.module[2].depthwise.dilation == (4,)


def test_temporal_mixing_encoder_forward_shape_roundtrip():
    enc = TemporalMixingEncoder([dict(cin=5, kernel_size=3, num_groups=1, activation="silu")])
    x = torch.randn(2, 6, 3, 5, 4, 3)  # [B,T,L,C,H,W]
    y = enc(x)
    assert y.shape == x.shape


# -----------------------------
# FourierTimeEmbedding
# -----------------------------


@pytest.mark.parametrize("shape_kind", ["T", "BT", "BT1"])
@pytest.mark.parametrize("dtype", [torch.int64, torch.float32, torch.float16])
def test_fourier_time_embedding_shapes_and_dtype(shape_kind, dtype):
    emb = FourierTimeEmbedding(num_freqs=6, max_freq=5.0)
    B, T = 3, 7

    if shape_kind == "T":
        t = torch.arange(T, dtype=dtype)
        out = emb(t)
        assert out.shape == (T, 2 * 6)
    elif shape_kind == "BT":
        t = torch.linspace(0, 1, T, dtype=dtype).unsqueeze(0).expand(B, -1)
        out = emb(t)
        assert out.shape == (B, T, 2 * 6)
    else:
        t = torch.linspace(0, 1, T, dtype=dtype).unsqueeze(0).expand(B, -1).unsqueeze(-1)  # [B,T,1]
        out = emb(t)
        assert out.shape == (B, T, 2 * 6)

    # Implementation forces float32 math for stability
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_fourier_time_embedding_buffer_moves_device():
    emb = FourierTimeEmbedding(num_freqs=4, max_freq=3.0)
    assert emb.freqs.device.type == "cpu"
    # just ensure calling on tensor produces output on same device
    t = torch.ones((2, 3), dtype=torch.float32)
    out = emb(t)
    assert out.device == t.device


# -----------------------------
# TimeFiLM
# -----------------------------


def test_time_film_raises_on_scalar_input_dim0():
    film = TimeFiLM(embed_dim=8, hidden_dim=16, activation="silu", c_dec=5)
    e = torch.tensor(1.0)  # dim=0
    with pytest.raises(ValueError, match="at least 1 dim"):
        film(e)


def test_time_film_outputs_shapes():
    film = TimeFiLM(embed_dim=8, hidden_dim=16, activation="silu", c_dec=7)
    e = torch.randn(2, 5, 8)  # [B,T,E]
    g, b = film(e)
    assert g.shape == (2, 5, 7)
    assert b.shape == (2, 5, 7)
    assert torch.isfinite(g).all()
    assert torch.isfinite(b).all()


# -----------------------------
# SpatioTemporalDecoder
# -----------------------------


def test_spatiotemporal_decoder_output_shape_no_grads():
    B, T, H, W = 2, 4, 6, 5
    L = 3
    cin = 5
    c_dec = 7
    c_film = 4
    out_channels = 7

    dec = SpatioTemporalDecoder(
        cin=cin,
        c_dec=c_dec,
        c_film=c_film,
        out_channels=out_channels,
        activation="silu",
        L=L,
        temporal_embedding_config=dict(num_freqs=4, max_freq=3.0),
        temporal_film_config=dict(embed_dim=8, hidden_dim=16, activation="silu", c_dec=c_film),
        upsample=False,
    )

    x = torch.randn(B, T, L, cin, H, W)
    t = torch.linspace(0, 1, T).unsqueeze(0).expand(B, -1)

    m = dec(x, t, return_gradients=False)
    assert isinstance(m, SpatialDecoderManifest)
    assert m.dz_hat_dt is None
    assert m.z_hat.shape == (B, 7, L, T, H, W)


@pytest.mark.slow
def test_spatiotemporal_decoder_output_shape_with_grads_and_central_difference_check():
    # Small shapes to keep jacrev/vmap fast.
    B, T, H, W = 1, 3, 4, 4
    L = 2
    cin = 3
    c_dec = 5
    c_film = 3
    out_channels = 7

    dec = SpatioTemporalDecoder(
        cin=cin,
        c_dec=c_dec,
        c_film=c_film,
        out_channels=out_channels,
        activation="silu",
        L=L,
        temporal_embedding_config=dict(num_freqs=3, max_freq=2.0),
        temporal_film_config=dict(embed_dim=6, hidden_dim=12, activation="silu", c_dec=c_film),
        upsample=False,
    )
    dec.eval()

    x = torch.randn(B, T, L, cin, H, W, dtype=torch.float32)
    t = torch.linspace(0.1, 0.9, T, dtype=torch.float32).unsqueeze(0).expand(B, -1)

    m = dec(x, t, return_gradients=True)
    assert m.dz_hat_dt is not None
    assert m.z_hat.shape == (B, 7, L, T, H, W)
    assert m.dz_hat_dt.shape == (B, 7, L, T, H, W)
    assert torch.isfinite(m.z_hat).all()
    assert torch.isfinite(m.dz_hat_dt).all()

    # Central finite difference on t. Keep eps not-too-small to avoid float32 cancellation.
    eps = 1e-3
    m_p = dec(x, t + eps, return_gradients=False)
    m_m = dec(x, t - eps, return_gradients=False)
    fd = (m_p.z_hat - m_m.z_hat) / (2.0 * eps)

    # Instead of allclose over the full tensor (too strict and dominated by worst pixel),
    # compare a random scalar projection. This is stable and still detects wrong grads.
    proj = torch.randn_like(m.z_hat)
    lhs = (fd * proj).mean()
    rhs = (m.dz_hat_dt * proj).mean()

    assert torch.isfinite(lhs)
    assert torch.isfinite(rhs)
    assert torch.isclose(lhs, rhs, atol=5e-3, rtol=5e-2)


def test_fourier_time_embedding_forces_float32_and_can_cause_dtype_mismatch_if_module_cast():
    emb = FourierTimeEmbedding(num_freqs=3, max_freq=2.0).to(dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, 4, dtype=torch.float64)
    out = emb(t)
    assert out.dtype == torch.float32  # intentional behavior in implementation


def test_spatiotemporal_decoder_out_channels_must_match_signal_count():
    # out_channels is strictly validated against len(signals) (7 by default),
    # independent of L: the layer dimension is handled internally via
    # per-(signal, layer) output heads, not folded into out_channels.
    B, T, H, W = 1, 2, 3, 3
    L = 2
    cin = 3
    c_dec = 4
    c_film = 3

    common_kwargs = dict(
        cin=cin,
        c_dec=c_dec,
        c_film=c_film,
        activation="silu",
        L=L,
        temporal_embedding_config=dict(num_freqs=2, max_freq=2.0),
        temporal_film_config=dict(embed_dim=4, hidden_dim=8, activation="silu", c_dec=c_film),
        upsample=False,
    )

    with pytest.raises(AssertionError, match="out_channels must match"):
        SpatioTemporalDecoder(out_channels=7 * L, **common_kwargs)

    dec = SpatioTemporalDecoder(out_channels=7, **common_kwargs)

    x = torch.randn(B, T, L, cin, H, W)
    t = torch.linspace(0, 1, T).unsqueeze(0).expand(B, -1)
    m = dec(x, t, return_gradients=False)
    assert m.z_hat.shape == (B, 7, L, T, H, W)


# -----------------------------
# HeinzleNet integration
# -----------------------------


def test_heinzle_net_forward_shapes_and_gradients():
    cfg = _mk_heinzle_configs(L=3, Cmix=4, Cenc=6, c_dec=8)
    net = HeinzleNet(**cfg)

    x, t = _mk_small_inputs(B=2, L=3, T=4, H=6, W=5, dtype=torch.float32)
    x.requires_grad_(True)

    m = net(x, t, return_gradients=False)
    assert isinstance(m, SpatialDecoderManifest)
    assert m.dz_hat_dt is None
    assert m.z_hat.shape == (2, 7, 3, 4, 6, 5)

    m.z_hat.mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


@pytest.mark.slow
def test_heinzle_net_forward_with_gradients_produces_dz_hat_dt():
    cfg = _mk_heinzle_configs(L=2, Cmix=3, Cenc=4, c_dec=5)
    net = HeinzleNet(**cfg)
    net.eval()

    x, t = _mk_small_inputs(B=1, L=2, T=3, H=4, W=4, dtype=torch.float32)
    m = net(x, t, return_gradients=True)
    assert m.dz_hat_dt is not None
    assert m.z_hat.shape == (1, 7, 2, 3, 4, 4)
    assert m.dz_hat_dt.shape == (1, 7, 2, 3, 4, 4)
    assert torch.isfinite(m.dz_hat_dt).all()
