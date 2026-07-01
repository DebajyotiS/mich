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
        self.expand_net = nn.Conv2d(1, self.C, kernel_size=1, bias=True)

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
        y = F.conv2d(x2d, W_eff, bias=self.b)  # [B*T, L, H, W]

        # Apply expand_net independently per layer: treat B*T*L as batch with cin=1
        y_btl = y.reshape(B * T * L, 1, H, W)
        y_exp = self.expand_net(y_btl)  # [B*T*L, C, H, W]
        y_exp = y_exp.view(B, T, L, self.C, H, W)
        return y_exp


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
        B, T, L, C, H, W = x.shape
        x = x.reshape(B * T * L, C, H, W)  # [B*T*L, C, H, W]
        for layer in self.module:
            x = checkpoint(layer, x, use_reentrant=False).to(x.dtype)
        _, C_out, H_out, W_out = x.shape
        x = x.view(B, T, L, C_out, H_out, W_out)  # [B, T, L, C', H, W]
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
        _pad = (kernel_size - 1) * dilation // 2  # unused, kept for reference

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
        x = F.pad(x, (pad // 2, pad - pad // 2))  # symmetric padding — non-causal
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
        B, T, L, C, H, W = x.shape
        x = x.permute(0, 2, 4, 5, 3, 1).reshape(B * L * H * W, C, T)  # [B*L*H*W, C, T]
        for layer in self.module:
            x = checkpoint(layer, x, use_reentrant=False).to(x.dtype)
        x = x.reshape(B, L, H, W, C, T).permute(0, 5, 1, 4, 2, 3).contiguous()  # [B, T, L, C, H, W]
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
                    if sig == "x":
                        head.bias.fill_(-3.0)
                    elif sig in ("f", "v", "q"):
                        head.bias.fill_(0.5413)
                    else:
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
        u, (B, T, L, H, W) = self._pre_film_features(x)
        gamma, beta = self._gamma_beta(t)

        z_pre = self._decode_from_film(u, gamma, beta, B, T, L, H, W)  # pre-activation
        z_hat = self._apply_activations(z_pre)  # post-activation

        if not return_gradients:
            return SpatialDecoderManifest(z_hat=z_hat)

        dgamma_dt, dbeta_dt = self._gamma_beta_time_grads(t)
        dz_pre_dt = self._decode_dt_from_film(u, dgamma_dt, dbeta_dt, B, T, L, H, W)

        # Chain rule: d/dt act(z) = act'(z) * dz/dt
        dz_hat_dt = self._apply_activation_derivatives(z_pre, dz_pre_dt)

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
        for l_idx in range(L):
            tok_l = layer_toks[l_idx]  # [layer_embed_dim]

            def _gb_from_scalar(
                ts: torch.Tensor,
                _tok_l: torch.Tensor = tok_l,
                _sig_toks: torch.Tensor = sig_toks,
            ) -> torch.Tensor:
                """ts: scalar -> [7, 2, c_film] = [gamma, beta] for all signals"""
                emb = self.time_embedding(ts)  # [2F]
                emb_exp = emb.unsqueeze(0).expand(N_SIG, -1)  # [7, 2F]
                tok_l_exp = _tok_l.unsqueeze(0).expand(N_SIG, -1).to(emb.dtype)
                film_in = torch.cat(
                    [emb_exp, tok_l_exp, _sig_toks.to(emb.dtype)], dim=-1
                )  # [7, 2F+layer_embed_dim+signal_embed_dim]
                g, b = self.time_film(film_in)  # [7, c_film]
                return torch.stack([g, b], dim=1)  # [7, 2, c_film]

            grads_flat = vmap(jacrev(_gb_from_scalar))(t_flat)  # [BT, 7, 2, c_film]
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

    def _decode_dt_from_film(self, u, dgamma_dt, dbeta_dt, B, T, L, H, W):
        # Same spatial projection as forward pass (u is constant w.r.t. t)
        u_film = self.signal_proj(u.reshape(B * T * L, -1, H, W))  # [B*T*L, c_film, H, W]
        c_film = u_film.shape[1]
        u_film = u_film.view(B, T, L, c_film, H, W)

        # Apply conv weights only -- bias terms are constant so their derivative is zero
        # Process one signal at a time to avoid materialising [B, T, L, 7, c_film, H, W]
        dout_parts = []
        for sig_idx in range(len(self.signals)):
            g_sig = dgamma_dt[:, :, :, sig_idx, :][..., None, None]  # [B, T, L, c_film, 1, 1]
            b_sig = dbeta_dt[:, :, :, sig_idx, :][..., None, None]  # [B, T, L, c_film, 1, 1]
            dy_sig = g_sig * u_film + b_sig  # [B, T, L, c_film, H, W]
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
        self.temporal_mixing = TemporalMixingEncoder(temporal_mixing_config)
        self.head = nn.Conv2d(c_enc, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: BOLD input [B, L, T, H, W]
        Returns:
            neural: predicted neural activity [B, L, T, H, W]
        """
        xmix = self.layer_mixing(x)        # [B, T, L, C, H, W]
        xenc = self.spatial_encoder(xmix)  # [B, T, L, C', H, W]
        xmix = self.temporal_mixing(xenc)  # [B, T, L, C', H, W]
        B, T, L, C, H, W = xmix.shape
        out = xmix.reshape(B * T * L, C, H, W)
        out = self.head(out)               # [B*T*L, 1, H, W]
        return out.reshape(B, L, T, H, W)


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
        xmix = self.layer_mixing(x)  # [B, T, L, C, H, W]
        xenc = self.spatial_encoder(xmix)  # [B, T, L, C', H, W]
        xmix = self.temporal_mixing(xenc)  # [B, T, L, C', H, W]
        z_hat = self.spatial_decoder(
            xmix, t, return_gradients=return_gradients
        )  # [B, 7, L, T, H, W]
        return z_hat
