"""HeinzleNet: the MICH encoder/decoder network, from raw BOLD to Heinzle latent states.

Pipeline (see `HeinzleNet.forward`): `MaskedLayerMixing` (mixes BOLD across adjacent
cortical layers, expands to C channels) -> `SpatialEncoder` (per-timestep 2D conv
stack) -> `TemporalMixingEncoder` (per-voxel temporal TCN) -> `SpatioTemporalDecoder`
(time-and-signal-conditioned FiLM decoder to the 7 Heinzle channels).

Axis-order convention: the encoder/temporal-mixing stack (`MaskedLayerMixing`,
`SpatialEncoder`, `TemporalMixingEncoder`) uses `[B, T, L, C, H, W]` (time before
layer). `SpatioTemporalDecoder`'s output, and everything downstream of it
(`SpatialDecoderManifest`, `mich.models.collocation`, `mich.models.mich_losses`),
uses `[B, C, L, T, H, W]` (channel-first, layer before time) instead. The decoder's
`forward` is where that flip happens; there is no shared symbolic name for both
orders because they never coexist in the same tensor.

Heinzle signal vocabulary: `HEINZLE_SIGNALS` (0=x, 1=s, 2=f, 3=v, 4=q, 5=vstar,
6=qstar) is the canonical channel ordering for every `[..., 7, ...]` tensor in this
module and in `mich.models.collocation`/`mich.models.mich_losses`. `x` is the neural
drive (forcing input, not itself governed by an ODE here); `s, f, v, q` are the
Heinzle/Balloon-Windkessel state variables; `vstar, qstar` are the delayed
inter-layer vascular-drainage terms, present only when a model is built with
`out_channels=7` (multi-layer/drain mode) -- `HEINZLE_SIGNALS_SINGLE` (5 channels,
no vstar/qstar) is used for single-layer/no-drain configs.
"""

from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping

import torch
from torch import nn
from torch.func import jacrev, vmap
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from mich.utils.torch_utils import (
    _softplus_deriv,
    get_activation,
)

HeinzleSignal = Literal["x", "s", "f", "v", "q", "vstar", "qstar"]
HEINZLE_SIGNALS: list[HeinzleSignal] = ["x", "s", "f", "v", "q", "vstar", "qstar"]
HEINZLE_SIGNALS_SINGLE: list[HeinzleSignal] = ["x", "s", "f", "v", "q"]
HEINZLE_N_SIGNALS = len(HEINZLE_SIGNALS)
HEINZLE_SIGNAL_IDX: dict[HeinzleSignal, int] = {s: i for i, s in enumerate(HEINZLE_SIGNALS)}


@dataclass
class SpatialDecoderManifest:
    """
    z_hat:     [B, 7, L, T, H, W]  -- post-activation Heinzle states
    dz_hat_dt: [B, 7, L, T, H, W]  -- d/dt of post-activation states (optional)

    Channel dim 1 follows HEINZLE_SIGNALS ordering:
        0=x, 1=s, 2=f, 3=v, 4=q, 5=vstar, 6=qstar
    """

    z_hat: torch.Tensor
    grads: torch.Tensor | None = None

    @property
    def dz_hat_dt(self) -> torch.Tensor | None:
        return self.grads if self.grads is not None else None

    def channel(self, signal: HeinzleSignal) -> torch.Tensor:
        """Return [B, L, T, H, W] slice for a named signal."""
        return self.z_hat[:, HEINZLE_SIGNAL_IDX[signal]]

    def channel_grad(self, signal: HeinzleSignal) -> torch.Tensor:
        """Return [B, L, T, H, W] time-derivative slice for a named signal."""
        if self.grads is None:
            raise RuntimeError("Gradients were not requested (return_gradients=False).")
        return self.grads[:, HEINZLE_SIGNAL_IDX[signal]]


@dataclass(frozen=True)
class ChannelActivation:
    """
    Elementwise activation + its analytic pointwise derivative.

    Both callables must accept and return tensors of arbitrary shape.
    """

    fn: Callable[[torch.Tensor], torch.Tensor]
    dfn_dx: Callable[[torch.Tensor], torch.Tensor]


_IDENTITY = ChannelActivation(
    fn=lambda x: x,
    dfn_dx=torch.ones_like,
)

HEINZLE_ACTIVATIONS: dict[HeinzleSignal, ChannelActivation] = {
    # identity, not softplus -- the neural-baseline generator (signals.py Pulse.generate,
    # baseline="random") injects signed offsets into the inter-pulse rest periods, and
    # nothing downstream requires x >= 0 (_sanitise_states clamps x to [-1e3, 1e3], the
    # same branch as s/vstar/qstar, not the f/v/q non-negative branch). softplus made
    # negative baselines architecturally unreachable.
    "x": _IDENTITY,
    "s": _IDENTITY,
    "f": ChannelActivation(fn=F.softplus, dfn_dx=_softplus_deriv),
    "v": ChannelActivation(fn=F.softplus, dfn_dx=_softplus_deriv),
    "q": ChannelActivation(fn=F.softplus, dfn_dx=_softplus_deriv),
    "vstar": _IDENTITY,
    "qstar": _IDENTITY,
}


HEINZLE_ACTIVATIONS_ORDERED: list[ChannelActivation] = [
    HEINZLE_ACTIVATIONS[s] for s in HEINZLE_SIGNALS
]


class MaskedLayerMixing(nn.Module):
    """Mix each cortical layer with the layer immediately below it, then expand
    1 -> C channels independently per layer.

    The 1x1 "conv2d" over the layer axis is restricted by `mask` (see
    `_generate_mask`) to only ever mix a layer with itself and the layer below,
    modelling one-directional vascular drainage/point-spread bleed-through
    between adjacent layers rather than an unrestricted L x L mixing matrix.

    Args:
        L: Number of cortical layers.
        C: Output channel count per layer after `expand_net`.
        init_identity: If True, initialise the per-layer mixing weight `W` to
            the identity (each layer starts as an unmixed copy of itself; the
            below-layer coupling is learned away from zero during training).
    """

    def __init__(self, L: int = 3, C: int = 16, init_identity: bool = True):
        super().__init__()
        self.C = int(C)
        self.L = int(L)

        self._generate_mask()

        self.W = nn.Parameter(torch.zeros((self.L, self.L, 1, 1)))  # fp32 params
        self.b = nn.Parameter(torch.zeros((self.L,)))  # fp32 params
        self.expand_net = nn.Conv2d(1, self.C, kernel_size=1, bias=True)

        if init_identity:
            with torch.no_grad():
                self.W.zero_()
                for i in range(self.L):
                    self.W[i, i, 0, 0] = 1.0

    def _generate_mask(self) -> None:
        """Register the `[L, L, 1, 1]` conv-weight mask: `mask[i, i] = 1` (self) and
        `mask[i, i-1] = 1` (the layer below), zero elsewhere -- multiplied elementwise
        against the learnable `W` in `forward` so gradient descent can never populate
        a masked-out (self, above, or non-adjacent) coupling."""
        mask = torch.zeros((self.L, self.L), dtype=torch.float32)
        idx = torch.arange(self.L)
        mask[idx, idx] = 1.0
        if self.L > 1:
            mask[idx[1:], idx[:-1]] = 1.0
        self.register_buffer("mask", mask.view(self.L, self.L, 1, 1), persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, T, H, W] BOLD (or any single-channel per-layer input).

        Returns:
            [B, T, L, C, H, W] -- masked layer mix, then per-layer 1 -> C expansion.

        Raises:
            AssertionError: If `x`'s layer count doesn't match `self.L`.
        """
        B, L, T, H, W = x.shape
        if L != self.L:
            raise AssertionError(f"Expected input with {self.L} layers, got {L}")

        x2d = x.permute(0, 2, 1, 3, 4).reshape(B * T, L, H, W)

        # No dtype casting. Autocast will handle mixed precision safely.
        W_eff = self.W * self.mask
        y = F.conv2d(x2d, W_eff, bias=self.b)  # [B*T, L, H, W]

        # Apply expand_net independently per layer: treat B*T*L as batch with cin=1
        y_btl = y.reshape(B * T * L, 1, H, W)
        y_exp = self.expand_net(y_btl)  # [B*T*L, C, H, W]
        y_exp = y_exp.view(B, T, L, self.C, H, W)
        return y_exp


class DepthWiseSeparableConvLayer(nn.Module):
    """Depthwise spatial conv -> pointwise channel mix -> GroupNorm -> activation."""

    def __init__(
        self,
        cin: int,
        cout: int,
        *,
        stride: int = 1,
        dw_kernel: int = 3,
        pw_kernel: int = 1,
        num_groups: int = 1,
        activation: str = "silu",
    ):
        super().__init__()

        self.depthwise = nn.Conv2d(
            cin,
            cin,
            kernel_size=dw_kernel,
            stride=stride,
            padding=(dw_kernel - 1) // 2,
            groups=cin,
            bias=False,
        )
        self.pointwise = nn.Conv2d(cin, cout, kernel_size=pw_kernel, bias=False)

        assert num_groups > 0 and cout % num_groups == 0, (
            "num_groups must be a positive divisor of cout"
        )
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=cout)
        self.activation = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class SpatialEncoder(nn.Module):
    """Stack of `DepthWiseSeparableConvLayer`s applied independently at every
    (batch, timestep, layer) -- i.e. purely spatial, no temporal or cross-layer
    mixing here (that happens in `TemporalMixingEncoder` and `MaskedLayerMixing`
    respectively). Each layer runs under `torch.utils.checkpoint` to trade
    recompute for activation memory.
    """

    def __init__(self, module_config: list[Mapping[str, Any]]):
        super().__init__()
        self.module = nn.ModuleList()
        for config in module_config:
            self.module.append(DepthWiseSeparableConvLayer(**config))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, L, C, H, W].

        Returns:
            [B, T, L, C', H, W] -- C' is the last layer's `cout`; H, W unchanged
            (every conv here is stride/padding-preserving).
        """
        B, T, L, C, H, W = x.shape
        x = x.reshape(B * T * L, C, H, W)  # [B*T*L, C, H, W]
        for layer in self.module:
            x = checkpoint(layer, x, use_reentrant=False).to(x.dtype)
        _, C_out, H_out, W_out = x.shape
        x = x.view(B, T, L, C_out, H_out, W_out)  # [B, T, L, C', H, W]
        return x


class TemporalDepthWiseTCNLayer(nn.Module):
    """Depthwise-separable 1D temporal conv with a residual connection.

    Warning:
        Padding is symmetric (`forward` pads both sides of the kernel equally),
        so this layer is non-causal: its output at timestep t can depend on
        input at t' > t. `dilation` (set by `TemporalMixingEncoder`) controls
        how far in both directions.
    """

    def __init__(
        self,
        cin: int,
        dilation: int = 1,
        kernel_size: int = 3,
        num_groups: int = 1,
        activation: str = "silu",
    ):
        super().__init__()

        self.depthwise = nn.Conv1d(
            cin,
            cin,
            kernel_size=kernel_size,
            padding=0,
            groups=cin,
            dilation=dilation,
            bias=False,
        )
        self.pointwise = nn.Conv1d(cin, cin, kernel_size=1, bias=False)

        assert num_groups > 0 and cin % num_groups == 0, (
            "num_groups must be a positive divisor of cin"
        )
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=cin)
        self.activation = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        pad = (self.depthwise.kernel_size[0] - 1) * self.depthwise.dilation[0]
        x = F.pad(x, (pad // 2, pad - pad // 2))  # symmetric padding — non-causal
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.activation(x)
        return x + residual


class TemporalMixingEncoder(nn.Module):
    """Stack of `TemporalDepthWiseTCNLayer`s applied independently per voxel
    (every (batch, layer, h, w) is its own length-T sequence).

    Args:
        module_config: Per-layer kwargs for `TemporalDepthWiseTCNLayer`.
        auto_dilation: If True, layer i's dilation is overridden to `2**i`
            (exponentially growing receptive field with depth), overwriting
            any `dilation` given in `module_config[i]`.
    """

    def __init__(self, module_config: list[Mapping[str, Any]], auto_dilation: bool = True):
        super().__init__()
        self.num_layers = len(module_config)
        self.module = nn.ModuleList()

        for i, config in enumerate(module_config):
            # Create a fresh dict copy to avoid mutating caller arguments
            layer_config = dict(config)
            if auto_dilation:
                layer_config["dilation"] = 2**i
            self.module.append(TemporalDepthWiseTCNLayer(**layer_config))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, L, C, H, W].

        Returns:
            [B, T, L, C, H, W] -- same shape; channel count is preserved by
            every `TemporalDepthWiseTCNLayer` (cin==cout, residual add).
        """
        B, T, L, C, H, W = x.shape

        # Flatten spatial and latent dimensions into batch dimension
        # Shape transition: [B, T, L, C, H, W] -> [B, L, H, W, C, T] -> [B*L*H*W, C, T]
        x = x.permute(0, 2, 4, 5, 3, 1).contiguous().view(B * L * H * W, C, T)

        for layer in self.module:
            x = checkpoint(layer, x, use_reentrant=False)

        # Restore original tensor dimensions
        # Shape transition: [B*L*H*W, C, T] -> [B, L, H, W, C, T] -> [B, T, L, C, H, W]
        x = x.view(B, L, H, W, C, T).permute(0, 5, 1, 4, 2, 3).contiguous()
        return x


class FourierTimeEmbedding(nn.Module):
    """Sinusoidal (NeRF-style) time embedding: log-spaced frequencies, each
    contributing a sin and a cos feature, for conditioning `TimeFiLM`.

    Args:
        num_freqs: Number of log-spaced frequencies between `min_freq` and
            `max_freq`; output width is `2 * num_freqs` (sin + cos per freq).
        min_freq, max_freq: Endpoints of the log-spaced frequency range.
    """

    def __init__(self, num_freqs: int = 16, min_freq: float = 0.1, max_freq: float = 10.0):
        super().__init__()
        self.num_freqs = int(num_freqs)
        self.min_freq = float(min_freq)
        self.max_freq = float(max_freq)

        # Precompute freqs in float32 on CPU; it'll move with the module to GPU.
        freqs = torch.logspace(
            start=torch.log10(torch.tensor(self.min_freq, dtype=torch.float32)),
            end=torch.log10(torch.tensor(self.max_freq, dtype=torch.float32)),
            steps=self.num_freqs,
            dtype=torch.float32,
        )  # [F]
        self.register_buffer("freqs", freqs, persistent=True)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B, T] (or [T], or [B, T, 1]).

        Returns:
            [B, T, 2F] (or [T, 2F] if input was [T]); F = `self.num_freqs`.

        Note:
            The trig computation runs in float32 with autocast disabled
            regardless of `t`'s input dtype/device autocast state, to avoid
            dtype-promotion surprises in the sin/cos ops; the output is
            float32 even under a bf16/fp16 autocast context.
        """
        # Squeeze trailing singleton dim if present
        if t.dim() >= 1 and t.shape[-1] == 1:
            t = t.squeeze(-1)

        # Ensure floating and run the embedding math in float32 for autocast safety.
        if not torch.is_floating_point(t):
            t = t.float()
        else:
            t = t.to(torch.float32)

        freqs = self.freqs.to(device=t.device, dtype=torch.float32)

        # Disable autocast for trig to avoid unexpected dtype promotion paths.
        device_type = "cuda" if t.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            ang = (2.0 * torch.pi) * t[..., None] * freqs  # [..., F]
            emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # [..., 2F]

        return emb


class TimeFiLM(nn.Module):
    """Feature-wise Linear Modulation (FiLM): maps a conditioning embedding to a
    per-channel (gamma, beta) pair, applied elsewhere as `gamma * features + beta`.

    Args:
        embed_dim: Width of the input conditioning embedding `e_t`.
        hidden_dim: Width of the single hidden layer between `embed_dim` and
            the `2 * c_dec`-wide (gamma, beta) output.
        activation: Name passed to `get_activation` for the hidden layer.
        c_dec: Width of the modulated feature space gamma/beta apply to.
    """

    def __init__(self, embed_dim: int, hidden_dim: int, activation: str, c_dec: int):
        super().__init__()
        self.c_dec = int(c_dec)
        self.linear = nn.Linear(embed_dim, hidden_dim)
        self.activation = get_activation(activation)
        self.out = nn.Linear(hidden_dim, 2 * self.c_dec)

    def forward(self, e_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            e_t: [..., E] conditioning embedding, E = `embed_dim`.

        Returns:
            (gamma, beta), each [..., c_dec].

        Raises:
            ValueError: If `e_t` has no dimensions (0-d).
        """
        if e_t.dim() < 1:
            raise ValueError(f"Expected e_t with at least 1 dim (E). Got shape {tuple(e_t.shape)}")

        orig_shape = e_t.shape[:-1]
        E = e_t.shape[-1]

        x = e_t.reshape(-1, E)  # [N, E]
        x = self.linear(x)  # [N, hidden]
        x = self.activation(x)
        x = self.out(x)  # [N, 2*c_dec]
        x = x.view(*orig_shape, 2, self.c_dec)  # [..., 2, c_dec]

        gamma = x[..., 0, :]  # [..., c_dec]
        beta = x[..., 1, :]  # [..., c_dec]
        return gamma, beta


class SpatioTemporalDecoder(nn.Module):
    """
    Spatial decoder with time-conditioned FiLM, optional per-pixel time
    derivatives, and per-channel output activations for Heinzle model states.

    FiLM is conditioned jointly on time, cortical layer, and signal identity,
    operating in a thin c_film-dimensional bottleneck projected from c_dec.
    This gives each of the 7 Heinzle signals its own dedicated temporal dynamics
    while keeping the total FiLM parameter budget comparable to the shared case.

    Inputs:
        x : [B, T, L, C_in, H, W]
        t : [B, T]

    Outputs (via SpatialDecoderManifest):
        z_hat     : [B, 7, L, T, H, W]  post-activation states
        dz_hat_dt : [B, 7, L, T, H, W]  d/dt of post-activation states
                    (only if return_gradients=True)

    Channel ordering: 0=x  1=s  2=f  3=v  4=q  5=v*  6=q*
    """

    def __init__(
        self,
        cin: int,
        c_dec: int,
        out_channels: int,
        activation: str,
        L: int,
        temporal_film_config: Mapping[str, Any],
        temporal_embedding_config: Mapping[str, Any],
        *,
        c_film: int,
        layer_embed_dim: int = 16,
        signal_embed_dim: int = 8,
        upsample: bool = False,
        signals: list[HeinzleSignal] = HEINZLE_SIGNALS,
    ):
        """
        Args:
            cin: Input channel count (from `TemporalMixingEncoder`).
            c_dec: Shared spatial-decoder width (`self.conv`'s output).
            out_channels: Must equal `len(signals)` (7, or 5 for
                `HEINZLE_SIGNALS_SINGLE`).
            activation: Name passed to `get_activation` for `self.conv`.
            L: Number of cortical layers; one output head per (signal, layer).
            temporal_film_config: Kwargs for `TimeFiLM`, minus `embed_dim`/
                `c_dec`, which this class computes and injects itself.
            temporal_embedding_config: Kwargs for `FourierTimeEmbedding`.
            c_film: Width of the FiLM bottleneck (`self.signal_proj`'s output),
                shared across signals before each signal's own output head.
            layer_embed_dim, signal_embed_dim: Widths of the learned layer- and
                signal-identity embeddings concatenated into the FiLM input.
            upsample: If True, 2x bilinear-upsample spatial features before
                `self.conv` (and update H, W accordingly).
            signals: Which Heinzle signals this decoder predicts, and in what
                channel order; must all be valid `HEINZLE_ACTIVATIONS` keys.

        Raises:
            AssertionError: If any `signals` entry isn't a valid Heinzle
                signal, or if `out_channels != len(signals)`.
        """
        super().__init__()

        assert all(s in HEINZLE_ACTIVATIONS for s in signals), (
            f"All signals must be valid Heinzle signals. Got: {signals}"
        )
        self.signals = signals
        self.signal_idx: dict[HeinzleSignal, int] = {s: i for i, s in enumerate(signals)}
        N_SIG = len(signals)
        assert out_channels == N_SIG, (
            f"out_channels must match len(signals): out_channels={out_channels}, len(signals)={N_SIG}"
        )
        channel_activations = [HEINZLE_ACTIVATIONS[s] for s in signals]
        self.L = L
        self.layer_embed_dim = layer_embed_dim
        self.signal_embed_dim = signal_embed_dim
        self.c_film = c_film
        self.upsample = upsample
        self.channel_activations = channel_activations  # None means identity everywhere

        # Shared spatial encoder: cin -> c_dec
        self.conv = DepthWiseSeparableConvLayer(
            cin=cin, cout=c_dec, activation=activation, stride=1
        )
        # Bottleneck projection: c_dec -> c_film (shared across signals)
        self.signal_proj = nn.Conv2d(c_dec, c_film, kernel_size=1, bias=False)
        # Per-(signal, layer) output heads: 7 * L independent Conv2d(c_film -> 1)
        # Indexed as head_idx = sig_idx * L + layer_idx
        self.out_heads = nn.ModuleList(
            [nn.Conv2d(c_film, 1, kernel_size=1, bias=True) for _ in range(N_SIG * L)]
        )
        with torch.no_grad():
            for sig_idx, sig in enumerate(signals):
                for layer_idx in range(L):
                    head = self.out_heads[sig_idx * L + layer_idx]
                    nn.init.zeros_(head.weight)
                    if sig in ("f", "v", "q"):
                        head.bias.fill_(0.5413)
                    else:
                        # x is now identity-activated (see HEINZLE_ACTIVATIONS), so a
                        # zero bias is an exact zero baseline with constant gradient=1
                        # -- no vanishing-gradient region to bootstrap out of.
                        nn.init.zeros_(head.bias)

        self.time_embedding = FourierTimeEmbedding(**temporal_embedding_config)
        self.layer_embed = nn.Embedding(L, layer_embed_dim)
        self.signal_embed = nn.Embedding(N_SIG, signal_embed_dim)

        num_freqs = temporal_embedding_config["num_freqs"]
        film_config = dict(temporal_film_config)
        film_config["embed_dim"] = 2 * num_freqs + layer_embed_dim + signal_embed_dim
        film_config["c_dec"] = c_film  # FiLM outputs c_film-dim gamma, beta per signal
        self.time_film = TimeFiLM(**film_config)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        return_gradients: bool = False,
    ) -> SpatialDecoderManifest:
        """
        Args:
            x: [B, T, L, C_in, H, W].
            t: [B, T], the [0, 1]-normalised time grid (see
                `CollocationMixin._make_time_grid`).
            return_gradients: If True, also compute `dz_hat_dt` (analytic d/dt
                of the post-activation states); otherwise `SpatialDecoderManifest.grads`
                is None and the whole derivative branch below is skipped.

        Returns:
            `SpatialDecoderManifest` with `z_hat`: [B, 7, L, T, H, W], and
            `grads` of the same shape if `return_gradients`, else None.

        Note:
            The derivative branch runs under `torch.autocast(..., enabled=False)`
            in float32 -- see the inline comment below for why a single
            precision boundary is used for the whole branch rather than only
            around the vmap/jacrev call.
        """
        u, (B, T, L, H, W) = self._pre_film_features(x)
        gamma, beta = self._gamma_beta(t)

        z_pre = self._decode_from_film(u, gamma, beta, B, T, L, H, W)  # pre-activation
        z_hat = self._apply_activations(z_pre)  # post-activation

        if not return_gradients:
            return SpatialDecoderManifest(z_hat=z_hat)

        # The whole analytic-derivative branch runs in float32, not just
        # _gamma_beta_time_grads's vmap(jacrev(...)) call -- mixing its (forced-fp32)
        # output with otherwise-bf16 tensors (u_film, gamma) here would create a new
        # fp32/bf16 seam that doesn't exist on the z_hat/value path, and composed
        # functorch transforms (vmap+jacrev) under autocast are fragile enough that
        # introducing a seam right at their boundary risks corrupting autocast's
        # dispatch state for unrelated ops later in the same forward pass. One clean
        # precision boundary for the entire branch avoids that class of bug entirely.
        device_type = "cuda" if t.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            u_fp32 = u.float()
            gamma_fp32 = gamma.float()
            dgamma_dt, dbeta_dt = self._gamma_beta_time_grads(t)
            # d(u)/d(t_norm): u is only sampled at the T recorded grid points (it comes
            # from temporal_mixing's TCN, which has no closed form in continuous t), so
            # this is a finite-difference estimate -- unlike dgamma_dt/dbeta_dt, which
            # are exact (jacrev). See _decode_dt_from_film for why this term is needed.
            du_dt = torch.gradient(u_fp32, dim=1)[0] * (T - 1)
            dz_pre_dt = self._decode_dt_from_film(
                u_fp32, du_dt, gamma_fp32, dgamma_dt, dbeta_dt, B, T, L, H, W
            )
            # Chain rule: d/dt act(z) = act'(z) * dz/dt
            dz_hat_dt = self._apply_activation_derivatives(z_pre.float(), dz_pre_dt)

        return SpatialDecoderManifest(z_hat=z_hat, grads=dz_hat_dt)

    def _pre_film_features(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int, int, int, int]]:
        """[B, T, L, C, H, W] -> u: [B, T, L, c_dec, H, W]"""
        B, T, L, C, H, W = x.shape
        x_btl = x.reshape(B * T * L, C, H, W)

        if self.upsample:
            x_btl = F.interpolate(x_btl, scale_factor=2, mode="bilinear", align_corners=False)
            _, _, H, W = x_btl.shape  # update H, W after upsample

        u_btl = checkpoint(self.conv, x_btl, use_reentrant=False).to(
            x_btl.dtype
        )  # [B*T*L, c_dec, H, W]
        c_dec = u_btl.shape[1]
        u = u_btl.view(B, T, L, c_dec, H, W)
        return u, (B, T, L, H, W)

    def _gamma_beta(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """t: [B, T] -> gamma, beta: [B, T, L, 7, c_film]"""
        B, T = t.shape
        L, N_SIG = self.L, len(self.signals)
        time_emb = self.time_embedding(t)  # [B, T, 2F]

        layer_ids = torch.arange(L, device=t.device)
        layer_toks = self.layer_embed(layer_ids)  # [L, layer_embed_dim]

        sig_ids = torch.arange(N_SIG, device=t.device)
        sig_toks = self.signal_embed(sig_ids)  # [7, signal_embed_dim]

        # Expand all to [B, T, L, 7, *] then concatenate
        time_emb_exp = time_emb[:, :, None, None, :].expand(B, T, L, N_SIG, -1)
        layer_toks_exp = layer_toks[None, None, :, None, :].expand(B, T, L, N_SIG, -1)
        sig_toks_exp = sig_toks[None, None, None, :, :].expand(B, T, L, N_SIG, -1)
        film_input = torch.cat(
            [time_emb_exp, layer_toks_exp, sig_toks_exp], dim=-1
        )  # [B, T, L, 7, 2F+layer_embed_dim+signal_embed_dim]

        gamma_flat, beta_flat = self.time_film(film_input.reshape(B * T * L * N_SIG, -1))
        c_film = gamma_flat.shape[-1]
        gamma = gamma_flat.view(B, T, L, N_SIG, c_film)
        beta = beta_flat.view(B, T, L, N_SIG, c_film)
        return gamma, beta

    def _gamma_beta_time_grads(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Analytic d(gamma)/dt and d(beta)/dt via vmap + jacrev over scalar t.
        Runs L separate vmap(jacrev(...)) calls to avoid tracing through nn.Embedding.
        Each call returns gradients for all 7 signals simultaneously.
        Returns: dgamma_dt, dbeta_dt -- each [B, T, L, 7, c_film]
        """
        B, T = t.shape
        L, N_SIG = self.L, len(self.signals)
        t_flat = t.reshape(-1)  # [BT]

        layer_ids = torch.arange(L, device=t.device)
        layer_toks = self.layer_embed(layer_ids)  # [L, layer_embed_dim] -- constant w.r.t. t

        sig_ids = torch.arange(N_SIG, device=t.device)
        sig_toks = self.signal_embed(sig_ids)  # [7, signal_embed_dim] -- constant w.r.t. t

        all_dgamma = []
        all_dbeta = []
        # vmap(jacrev(...)) is incompatible with autocast -- the composed functorch
        # transform's wrapper tensors hit "Unexpected floating ScalarType in
        # at::autocast::prioritize" under bf16-mixed. Same class of issue as
        # TimeEmbedding.forward's trig ops above; disable autocast for the whole
        # vmap/jacrev call and run it in float32.
        device_type = "cuda" if t.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            for l_idx in range(L):
                tok_l = layer_toks[l_idx].float()  # [layer_embed_dim]

                def _gb_from_scalar(
                    ts: torch.Tensor,
                    _tok_l: torch.Tensor = tok_l,
                    _sig_toks: torch.Tensor = sig_toks,
                ) -> torch.Tensor:
                    """ts: scalar -> [7, 2, c_film] = [gamma, beta] for all signals"""
                    emb = self.time_embedding(ts).float()  # [2F]
                    emb_exp = emb.unsqueeze(0).expand(N_SIG, -1)  # [7, 2F]
                    tok_l_exp = _tok_l.unsqueeze(0).expand(N_SIG, -1).to(emb.dtype)
                    film_in = torch.cat(
                        [emb_exp, tok_l_exp, _sig_toks.to(emb.dtype)], dim=-1
                    )  # [7, 2F+layer_embed_dim+signal_embed_dim]
                    g, b = self.time_film(film_in.float())  # [7, c_film]
                    return torch.stack([g, b], dim=1)  # [7, 2, c_film]

                grads_flat = vmap(jacrev(_gb_from_scalar))(t_flat.float())  # [BT, 7, 2, c_film]
                c_film = grads_flat.shape[-1]
                grads = grads_flat.view(B, T, N_SIG, 2, c_film)
                all_dgamma.append(grads[:, :, :, 0, :])  # [B, T, 7, c_film]
                all_dbeta.append(grads[:, :, :, 1, :])  # [B, T, 7, c_film]

            dgamma_dt = torch.stack(all_dgamma, dim=2)  # [B, T, L, 7, c_film]
            dbeta_dt = torch.stack(all_dbeta, dim=2)  # [B, T, L, 7, c_film]
        return dgamma_dt, dbeta_dt

    def _decode_from_film(
        self,
        u: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        B: int,
        T: int,
        L: int,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """Project -> per-signal FiLM -> per-(signal,layer) head -> [B, 7, L, T, H, W]  (pre-activation)"""
        # Project shared spatial features to FiLM bottleneck: c_dec -> c_film
        u_film = checkpoint(
            self.signal_proj, u.reshape(B * T * L, -1, H, W), use_reentrant=False
        ).to(u.dtype)  # [B*T*L, c_film, H, W]
        c_film = u_film.shape[1]
        u_film = u_film.view(B, T, L, c_film, H, W)

        # Per-signal FiLM: process one signal at a time to avoid [B, T, L, 7, c_film, H, W]
        out_parts = []
        for sig_idx in range(len(self.signals)):
            g_sig = gamma[:, :, :, sig_idx, :][..., None, None]  # [B, T, L, c_film, 1, 1]
            b_sig = beta[:, :, :, sig_idx, :][..., None, None]  # [B, T, L, c_film, 1, 1]
            y_sig = g_sig * u_film + b_sig  # [B, T, L, c_film, H, W]
            layer_parts = []
            for layer_idx in range(L):
                head = self.out_heads[sig_idx * L + layer_idx]
                y_sl = y_sig[:, :, layer_idx].reshape(B * T, self.c_film, H, W)
                out_sl = head(y_sl)  # [B*T, 1, H, W]
                layer_parts.append(out_sl.view(B, T, 1, 1, H, W))
            out_parts.append(torch.cat(layer_parts, dim=2))  # [B, T, L, 1, H, W]
        out = torch.cat(out_parts, dim=3)  # [B, T, L, 7, H, W]
        return out.permute(0, 3, 2, 1, 4, 5).contiguous()  # [B, 7, L, T, H, W]

    def _decode_dt_from_film(self, u, du_dt, gamma, dgamma_dt, dbeta_dt, B, T, L, H, W):
        """Total derivative of z_pre = u_film * gamma(t) + beta(t) w.r.t. t_norm, via the
        product rule: d(u_film)/dt * gamma + u_film * dgamma/dt + dbeta/dt. The previous
        version computed only the last two terms, treating u_film as constant w.r.t. t --
        exact for the FiLM branch alone, but structurally blind to u's own time-dependence,
        which is where temporal_mixing's TCN actually carries the signal's dynamics (the O1
        ablation showed replacing u with its temporal mean collapses this operator gap but
        wrecks value fit -- the dynamics really do route through u).

        signal_proj is a per-timestep-shared, bias-free linear map, so it commutes exactly
        with a time-derivative: signal_proj(du_dt) really is d(signal_proj(u))/dt, not an
        approximation on top of du_dt's own (finite-difference) one.
        """
        u_film = self.signal_proj(u.reshape(B * T * L, -1, H, W))  # [B*T*L, c_film, H, W]
        c_film = u_film.shape[1]
        u_film = u_film.view(B, T, L, c_film, H, W)

        du_film_dt = self.signal_proj(du_dt.reshape(B * T * L, -1, H, W))
        du_film_dt = du_film_dt.view(B, T, L, c_film, H, W)

        # Apply conv weights only -- bias terms are constant so their derivative is zero
        # Process one signal at a time to avoid materialising [B, T, L, 7, c_film, H, W]
        dout_parts = []
        for sig_idx in range(len(self.signals)):
            gamma_sig = gamma[:, :, :, sig_idx, :][..., None, None]  # [B, T, L, c_film, 1, 1]
            g_sig = dgamma_dt[:, :, :, sig_idx, :][..., None, None]  # [B, T, L, c_film, 1, 1]
            b_sig = dbeta_dt[:, :, :, sig_idx, :][..., None, None]  # [B, T, L, c_film, 1, 1]
            dy_sig = du_film_dt * gamma_sig + g_sig * u_film + b_sig  # [B, T, L, c_film, H, W]
            layer_parts = []
            for layer_idx in range(L):
                head = self.out_heads[sig_idx * L + layer_idx]
                dy_sl = dy_sig[:, :, layer_idx].reshape(B * T, self.c_film, H, W)
                dout_sl = F.conv2d(
                    dy_sl,
                    head.weight,
                    bias=None,
                    stride=head.stride,
                    padding=head.padding,
                )  # [B*T, 1, H, W]
                layer_parts.append(dout_sl.view(B, T, 1, 1, H, W))
            dout_parts.append(torch.cat(layer_parts, dim=2))  # [B, T, L, 1, H, W]
        dout = torch.cat(dout_parts, dim=3)  # [B, T, L, 7, H, W]
        return dout.permute(0, 3, 2, 1, 4, 5).contiguous()  # [B, 7, L, T, H, W]

    def _apply_activations(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B, 7, L, T, H, W]
        Applies channel_activations[i].fn to z[:, i] for each signal i.
        Returns tensor of same shape. No-op if channel_activations is None.
        """
        if self.channel_activations is None:
            return z
        return torch.stack(
            [self.channel_activations[i].fn(z[:, i]) for i in range(len(self.signals))],
            dim=1,
        )

    def _apply_activation_derivatives(
        self,
        z_pre: torch.Tensor,  # pre-activation [B, 7, L, T, H, W]
        dz_dt: torch.Tensor,  # pre-activation time derivative, same shape
    ) -> torch.Tensor:
        """
        Chain rule: d/dt act(z) = act'(z) * dz/dt  (elementwise per channel).
        Returns tensor of same shape. No-op if channel_activations is None.
        """
        if self.channel_activations is None:
            return dz_dt
        return torch.stack(
            [
                self.channel_activations[i].dfn_dx(z_pre[:, i]) * dz_dt[:, i]
                for i in range(len(self.signals))
            ],
            dim=1,
        )


class FullySupervisedNet(nn.Module):
    """Encoder-only baseline: BOLD → neural (no physics, no decoder).

    Shares the same spatial encoder and temporal TCN as HeinzleNet but
    replaces the SpatioTemporalDecoder with a single 1×1 conv head that
    directly regresses neural activity at every voxel and timestep.
    """

    def __init__(
        self,
        layer_mixing_config: Mapping[str, Any],
        spatial_encoder_config: list[Mapping[str, Any]],
        temporal_mixing_config: list[Mapping[str, Any]],
        c_enc: int,
    ):
        super().__init__()
        self.layer_mixing = MaskedLayerMixing(**layer_mixing_config)
        self.spatial_encoder = SpatialEncoder(spatial_encoder_config)
        self.temporal_mixing = TemporalMixingEncoder(temporal_mixing_config, auto_dilation=True)
        self.head = nn.Conv2d(c_enc, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: BOLD input [B, L, T, H, W]
        Returns:
            neural: predicted neural activity [B, L, T, H, W]
        """
        xmix = self.layer_mixing(x)  # [B, T, L, C, H, W]
        xenc = self.spatial_encoder(xmix)  # [B, T, L, C', H, W]
        xmix = self.temporal_mixing(xenc)  # [B, T, L, C', H, W]
        B, T, L, C, H, W = xmix.shape
        out = xmix.reshape(B * T * L, C, H, W)
        out = self.head(out)  # [B*T*L, 1, H, W]
        return out.reshape(B, L, T, H, W)


class HeinzleNet(nn.Module):
    """Full BOLD -> Heinzle-states network: see the module docstring's pipeline
    summary. `mich.models.mich.MICH` wraps one instance of this as `heinzle_net`.

    Args:
        layer_mixing_config: Kwargs for `MaskedLayerMixing`.
        spatial_encoder_config: Per-layer kwargs list for `SpatialEncoder`.
        temporal_mixing_config: Per-layer kwargs list for `TemporalMixingEncoder`.
        time_embedding_config: Kwargs for `FourierTimeEmbedding`, forwarded
            into `SpatioTemporalDecoder` as `temporal_embedding_config`.
        time_film_config: Kwargs for `TimeFiLM`, forwarded into
            `SpatioTemporalDecoder` as `temporal_film_config`.
        spatial_decoder_config: Remaining `SpatioTemporalDecoder` kwargs.
        auto_dilation: Forwarded to `TemporalMixingEncoder`.
    """

    def __init__(
        self,
        layer_mixing_config: Mapping[str, Any],
        spatial_encoder_config: list[Mapping[str, Any]],
        temporal_mixing_config: list[Mapping[str, Any]],
        time_embedding_config: Mapping[str, Any],
        time_film_config: Mapping[str, Any],
        spatial_decoder_config: Mapping[str, Any],
        auto_dilation: bool = False,
    ):
        super().__init__()
        self.layer_mixing = MaskedLayerMixing(**layer_mixing_config)
        self.spatial_encoder = SpatialEncoder(spatial_encoder_config)
        self.temporal_mixing = TemporalMixingEncoder(
            temporal_mixing_config, auto_dilation=auto_dilation
        )
        self.spatial_decoder = SpatioTemporalDecoder(
            **spatial_decoder_config,
            temporal_embedding_config=time_embedding_config,
            temporal_film_config=time_film_config,
        )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, return_gradients: bool = False
    ) -> SpatialDecoderManifest:
        """
        Args:
            x: [B, L, T, H, W] BOLD input.
            t: [B, T], the [0, 1]-normalised time grid.
            return_gradients: Forwarded to `SpatioTemporalDecoder.forward`.

        Returns:
            `SpatialDecoderManifest` with `z_hat`: [B, 7, L, T, H, W] (and
            `grads` of the same shape if `return_gradients`).
        """
        xmix = self.layer_mixing(x)  # [B, T, L, C, H, W]
        xenc = self.spatial_encoder(xmix)  # [B, T, L, C', H, W]
        xmix = self.temporal_mixing(xenc)  # [B, T, L, C', H, W]
        z_hat = self.spatial_decoder(
            xmix, t, return_gradients=return_gradients
        )  # [B, 7, L, T, H, W]
        return z_hat
