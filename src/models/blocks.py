from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping

import torch
from torch import nn
from torch.func import jacrev, vmap
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from src.utils.torch_utils import (
    _softplus_deriv,
    get_activation,
)

HeinzleSignal = Literal["x", "s", "f", "v", "q", "vstar", "qstar"]
HEINZLE_SIGNALS: list[HeinzleSignal] = ["x", "s", "f", "v", "q", "vstar", "qstar"]
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
    "x": ChannelActivation(fn=F.softplus, dfn_dx=_softplus_deriv),
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


def _init_heinzle_output_bias(out_conv: nn.Conv2d, L: int) -> None:
    if out_conv.bias is None:
        raise ValueError("Expected out_conv to have a bias for Heinzle output initialization.")
    if out_conv.out_channels != 7:
        raise ValueError(
            f"Expected out_conv to have out_channels=7 for Heinzle output initialization, "
            f"got {out_conv.out_channels}."
        )

    softplus_inv_1 = 0.5413  # softplus(0.5413)
    softplus_inv_0 = -3.0  # softplus(-3.0)

    x_idx = HEINZLE_SIGNAL_IDX["x"]
    f_idx = HEINZLE_SIGNAL_IDX["f"]
    v_idx = HEINZLE_SIGNAL_IDX["v"]
    q_idx = HEINZLE_SIGNAL_IDX["q"]

    with torch.no_grad():
        out_conv.bias.zero_()
        out_conv.bias[x_idx] = softplus_inv_0
        for sidx in (f_idx, v_idx, q_idx):
            out_conv.bias[sidx] = softplus_inv_1


class MaskedLayerMixing(nn.Module):
    def __init__(self, L: int = 3, C: int = 16, init_identity: bool = True):
        super().__init__()
        self.C = int(C)
        self.L = int(L)

        self._generate_mask()

        self.W = nn.Parameter(torch.zeros((self.L, self.L, 1, 1)))  # fp32 params
        self.b = nn.Parameter(torch.zeros((self.L,)))  # fp32 params
        self.expand_net = nn.Conv2d(self.L, self.C, kernel_size=1, bias=True)

        if init_identity:
            with torch.no_grad():
                self.W.zero_()
                for i in range(self.L):
                    self.W[i, i, 0, 0] = 1.0

    def _generate_mask(self) -> None:
        mask = torch.zeros((self.L, self.L), dtype=torch.float32)
        idx = torch.arange(self.L)
        mask[idx, idx] = 1.0
        if self.L > 1:
            mask[idx[1:], idx[:-1]] = 1.0
        self.register_buffer("mask", mask.view(self.L, self.L, 1, 1), persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, T, H, W = x.shape
        if L != self.L:
            raise AssertionError(f"Expected input with {self.L} layers, got {L}")

        x2d = x.permute(0, 2, 1, 3, 4).reshape(B * T, L, H, W)

        # No dtype casting. Autocast will handle mixed precision safely.
        W_eff = self.W * self.mask
        y = F.conv2d(x2d, W_eff, bias=self.b)
        y = self.expand_net(y)
        y = y.view(B, T, self.C, H, W)
        return y


class DepthWiseSeparableConvLayer(nn.Module):
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

        assert (
            num_groups > 0 and cout % num_groups == 0
        ), "num_groups must be a positive divisor of cout"
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=cout)
        self.activation = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class SpatialEncoder(nn.Module):
    def __init__(self, module_config: list[Mapping[str, Any]]):
        super().__init__()
        self.module = nn.ModuleList()
        for config in module_config:
            self.module.append(DepthWiseSeparableConvLayer(**config))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)  # [B*T, C, H, W]
        for layer in self.module:
            x = checkpoint(layer, x, use_reentrant=False).to(x.dtype)
        _, C_out, H_out, W_out = x.shape
        x = x.view(B, T, C_out, H_out, W_out)  # [B, T, C', H, W]
        return x


class TemporalDepthWiseTCNLayer(nn.Module):
    def __init__(
        self,
        cin: int,
        dilation: int = 1,
        kernel_size: int = 3,
        num_groups: int = 1,
        activation: str = "silu",
    ):
        super().__init__()
        # This pad is no longer used. We do explicit left-padding in forward() to ensure causality.
        _pad = (kernel_size - 1) * dilation // 2

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

        assert (
            num_groups > 0 and cin % num_groups == 0
        ), "num_groups must be a positive divisor of cin"
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=cin)
        self.activation = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        pad = (self.depthwise.kernel_size[0] - 1) * self.depthwise.dilation[0]
        x = F.pad(x, (pad, 0))  # left-pad for causality
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.activation(x)
        return x + residual


class TemporalMixingEncoder(nn.Module):
    def __init__(self, module_config: list[Mapping[str, Any]]):
        super().__init__()
        self.num_layers = len(module_config)
        self.module = nn.ModuleList()
        dilations = [2**i for i in range(self.num_layers)]
        for i, config in enumerate(module_config):
            config = dict(config)
            config["dilation"] = dilations[i]
            self.module.append(TemporalDepthWiseTCNLayer(**config))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x = x.permute(0, 3, 4, 2, 1).reshape(B * H * W, C, T)  # [B*H*W, C, T]
        for layer in self.module:
            x = checkpoint(layer, x, use_reentrant=False).to(x.dtype)
        x = x.reshape(B, H, W, C, T).permute(0, 4, 3, 1, 2).contiguous()
        return x


class FourierTimeEmbedding(nn.Module):
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
        t: [B, T] (or [T], or [B,T,1])
        returns: [B, T, 2F] (or [T, 2F] if input was [T])
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
    def __init__(self, embed_dim: int, hidden_dim: int, activation: str, c_dec: int):
        super().__init__()
        self.c_dec = int(c_dec)
        self.linear = nn.Linear(embed_dim, hidden_dim)
        self.activation = get_activation(activation)
        self.out = nn.Linear(hidden_dim, 2 * self.c_dec)

    def forward(self, e_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        e_t: [..., E]
        returns gamma, beta: [..., c_dec]
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

    Inputs:
        x : [B, T, C_in, H, W]
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
        layer_embed_dim: int = 16,
        upsample: bool = False,
        channel_activations: list[ChannelActivation] | None = HEINZLE_ACTIVATIONS_ORDERED,
    ):
        super().__init__()

        if channel_activations is not None:
            assert len(channel_activations) == HEINZLE_N_SIGNALS, (
                f"channel_activations must have {HEINZLE_N_SIGNALS} entries "
                f"(one per Heinzle signal), got {len(channel_activations)}."
            )

        self.L = L
        self.layer_embed_dim = layer_embed_dim
        self.upsample = upsample
        self.channel_activations = channel_activations  # None means identity everywhere

        self.conv = DepthWiseSeparableConvLayer(
            cin=cin, cout=c_dec, activation=activation, stride=1
        )
        self.out = nn.Conv2d(c_dec, 7, kernel_size=1)
        self.time_embedding = FourierTimeEmbedding(**temporal_embedding_config)
        self.layer_embed = nn.Embedding(L, layer_embed_dim)

        num_freqs = temporal_embedding_config["num_freqs"]
        film_config = dict(temporal_film_config)
        film_config["embed_dim"] = 2 * num_freqs + layer_embed_dim
        self.time_film = TimeFiLM(**film_config)

        if self.channel_activations is not None:
            _init_heinzle_output_bias(self.out, self.L)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        return_gradients: bool = False,
    ) -> SpatialDecoderManifest:
        u, (B, T, H, W) = self._pre_film_features(x)
        gamma, beta = self._gamma_beta(t)

        z_pre = self._decode_from_film(u, gamma, beta, B, T, H, W)  # pre-activation
        z_hat = self._apply_activations(z_pre)  # post-activation

        if not return_gradients:
            return SpatialDecoderManifest(z_hat=z_hat)

        dgamma_dt, dbeta_dt = self._gamma_beta_time_grads(t)
        dz_pre_dt = self._decode_dt_from_film(u, dgamma_dt, dbeta_dt, B, T, H, W)

        # Chain rule: d/dt act(z) = act'(z) * dz/dt
        dz_hat_dt = self._apply_activation_derivatives(z_pre, dz_pre_dt)

        return SpatialDecoderManifest(z_hat=z_hat, grads=dz_hat_dt)

    def _pre_film_features(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        """[B, T, C, H, W] -> u: [B, T, c_dec, H, W]"""
        B, T, C, H, W = x.shape
        x_bt = x.reshape(B * T, C, H, W)

        if self.upsample:
            x_bt = F.interpolate(x_bt, scale_factor=2, mode="bilinear", align_corners=False)
            _, _, H, W = x_bt.shape  # update H, W after upsample

        u_bt = self.conv(x_bt)  # [BT, c_dec, H, W]
        c_dec = u_bt.shape[1]
        u = u_bt.view(B, T, c_dec, H, W)
        return u, (B, T, H, W)

    def _gamma_beta(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """t: [B, T] -> gamma, beta: [B, T, L, c_dec]"""
        B, T = t.shape
        L = self.L
        time_emb = self.time_embedding(t)  # [B, T, 2F]

        layer_ids = torch.arange(L, device=t.device)
        layer_toks = self.layer_embed(layer_ids)  # [L, layer_embed_dim]

        time_emb_exp = time_emb.unsqueeze(2).expand(B, T, L, -1)  # [B, T, L, 2F]
        layer_toks_exp = layer_toks[None, None, :, :].expand(
            B, T, L, -1
        )  # [B, T, L, layer_embed_dim]
        film_input = torch.cat(
            [time_emb_exp, layer_toks_exp], dim=-1
        )  # [B, T, L, 2F+layer_embed_dim]
        film_input_flat = film_input.reshape(B * T * L, -1)

        gamma_flat, beta_flat = self.time_film(film_input_flat)  # [B*T*L, c_dec]
        c_dec = gamma_flat.shape[-1]
        gamma = gamma_flat.view(B, T, L, c_dec)
        beta = beta_flat.view(B, T, L, c_dec)
        return gamma, beta

    def _gamma_beta_time_grads(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Analytic d(gamma)/dt and d(beta)/dt via vmap + jacrev over scalar t.
        Runs L separate vmap(jacrev(...)) calls to avoid tracing through nn.Embedding.
        Returns: dgamma_dt, dbeta_dt -- each [B, T, L, c_dec]
        """
        B, T = t.shape
        L = self.L
        t_flat = t.reshape(-1)  # [BT]

        layer_ids = torch.arange(L, device=t.device)
        layer_toks = self.layer_embed(layer_ids)  # [L, layer_embed_dim] -- constant w.r.t. t

        all_dgamma = []
        all_dbeta = []
        for l_idx in range(L):
            tok = layer_toks[l_idx]  # [layer_embed_dim]

            def _gb_from_scalar(ts: torch.Tensor, _tok: torch.Tensor = tok) -> torch.Tensor:
                """ts: scalar -> [2, c_dec] = [gamma; beta]"""
                emb = self.time_embedding(ts)  # [2F]
                film_in = torch.cat([emb, _tok.to(emb.dtype)], dim=-1)  # [2F+layer_embed_dim]
                g, b = self.time_film(film_in)
                return torch.stack([g, b], dim=0)  # [2, c_dec]

            grads_flat = vmap(jacrev(_gb_from_scalar))(t_flat)  # [BT, 2, c_dec]
            c_dec = grads_flat.shape[-1]
            grads = grads_flat.view(B, T, 2, c_dec)
            all_dgamma.append(grads[:, :, 0, :])  # [B, T, c_dec]
            all_dbeta.append(grads[:, :, 1, :])  # [B, T, c_dec]

        dgamma_dt = torch.stack(all_dgamma, dim=2)  # [B, T, L, c_dec]
        dbeta_dt = torch.stack(all_dbeta, dim=2)  # [B, T, L, c_dec]
        return dgamma_dt, dbeta_dt

    def _decode_from_film(
        self,
        u: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        B: int,
        T: int,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """FiLM -> 1x1 conv -> [B, 7, L, T, H, W]  (pre-activation)"""
        L = self.L
        c_dec = u.shape[2]

        # u: [B, T, c_dec, H, W] -> [B, T, L, c_dec, H, W]
        u_exp = u.unsqueeze(2).expand(B, T, L, c_dec, H, W)
        g = gamma[..., None, None]  # [B, T, L, c_dec, 1, 1]
        b = beta[..., None, None]  # [B, T, L, c_dec, 1, 1]
        y = g * u_exp + b  # [B, T, L, c_dec, H, W]

        y_btl = y.reshape(B * T * L, c_dec, H, W)
        out_btl = self.out(y_btl)  # [B*T*L, 7, H, W]
        return out_btl.view(B, T, L, 7, H, W).permute(0, 3, 2, 1, 4, 5).contiguous()

    def _decode_dt_from_film(self, u, dgamma_dt, dbeta_dt, B, T, H, W):
        L = self.L
        c_dec = u.shape[2]

        u_exp = u.unsqueeze(2).expand(B, T, L, c_dec, H, W)
        g = dgamma_dt[..., None, None]  # [B, T, L, c_dec, 1, 1]
        b = dbeta_dt[..., None, None]  # [B, T, L, c_dec, 1, 1]
        dy = g * u_exp + b  # [B, T, L, c_dec, H, W]

        dy_btl = dy.reshape(B * T * L, c_dec, H, W)
        # Apply conv weights only -- no bias for derivative
        dout_btl = F.conv2d(
            dy_btl,
            self.out.weight,
            bias=None,
            stride=self.out.stride,
            padding=self.out.padding,
        )  # [B*T*L, 7, H, W]
        return dout_btl.view(B, T, L, 7, H, W).permute(0, 3, 2, 1, 4, 5).contiguous()

    def _apply_activations(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B, 7, L, T, H, W]
        Applies channel_activations[i].fn to z[:, i] for each signal i.
        Returns tensor of same shape. No-op if channel_activations is None.
        """
        if self.channel_activations is None:
            return z
        return torch.stack(
            [self.channel_activations[i].fn(z[:, i]) for i in range(HEINZLE_N_SIGNALS)],
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
                for i in range(HEINZLE_N_SIGNALS)
            ],
            dim=1,
        )


class HeinzleNet(nn.Module):
    def __init__(
        self,
        layer_mixing_config: Mapping[str, Any],
        spatial_encoder_config: list[Mapping[str, Any]],
        temporal_mixing_config: list[Mapping[str, Any]],
        time_embedding_config: Mapping[str, Any],
        time_film_config: Mapping[str, Any],
        spatial_decoder_config: Mapping[str, Any],
    ):
        super().__init__()
        self.layer_mixing = MaskedLayerMixing(**layer_mixing_config)
        self.spatial_encoder = SpatialEncoder(spatial_encoder_config)
        self.temporal_mixing = TemporalMixingEncoder(temporal_mixing_config)
        self.spatial_decoder = SpatioTemporalDecoder(
            **spatial_decoder_config,
            temporal_embedding_config=time_embedding_config,
            temporal_film_config=time_film_config,
        )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, return_gradients: bool = False
    ) -> SpatialDecoderManifest:
        xmix = self.layer_mixing(x)  # [B, T, C, H, W]
        xenc = self.spatial_encoder(xmix)  # [B, T, C', H, W]
        xmix = self.temporal_mixing(xenc)  # [B, T, C', H, W]
        z_hat = self.spatial_decoder(
            xmix, t, return_gradients=return_gradients
        )  # [B, 7, L, T, H, W]
        return z_hat
