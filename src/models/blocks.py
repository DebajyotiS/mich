from dataclasses import dataclass
from typing import Any, Literal, Mapping

import torch
from torch import nn
from torch.func import jacrev, vmap
from torch.nn import functional as F

from src.utils.torch_utils import get_activation

HeinzleSignal = Literal["x", "s", "f", "v", "q", "vstar", "qstar"]


@dataclass(frozen=True)
class SpatialDecoderManifest:
    z_hat: torch.Tensor  # [B, 7, L, T, H, W]
    dz_hat_dt: torch.Tensor | None = None


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
    def __init__(self, module_config: list[Mapping[str, Any]]):
        super().__init__()
        self.module = nn.ModuleList()
        for config in module_config:
            self.module.append(DepthWiseSeparableConvLayer(**config))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)  # [B*T, C, H, W]
        for layer in self.module:
            x = layer(x)
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
        pad = (kernel_size - 1) * dilation // 2

        self.depthwise = nn.Conv1d(
            cin,
            cin,
            kernel_size=kernel_size,
            padding=pad,
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
            x = layer(x)

        x = x.reshape(B, H, W, C, T).permute(0, 4, 3, 1, 2).contiguous()
        return x


class FourierTimeEmbedding(nn.Module):
    def __init__(self, num_freqs: int = 16, max_freq: float = 10.0):
        super().__init__()
        self.num_freqs = int(num_freqs)
        self.max_freq = float(max_freq)

        # Precompute freqs in float32 on CPU; it'll move with the module to GPU.
        freqs = torch.logspace(
            start=0.0,
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
    Spatial decoder with time-conditioned FiLM and optional per-pixel time derivatives.

    Inputs:
    x: [B, T, C_in, H, W]
    t: [B, T]

    Outputs:
    z_hat: [B, 7, L, T, H, W]
    optional grads: DecoderGrads with dz_hat_dt of same shape
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
        upsample: bool = False,
    ):
        super().__init__()
        self.L = L
        self.upsample = upsample

        self.conv = DepthWiseSeparableConvLayer(
            cin=cin, cout=c_dec, activation=activation, stride=1
        )
        self.out = nn.Conv2d(c_dec, out_channels, kernel_size=1)

        self.time_embedding = FourierTimeEmbedding(**temporal_embedding_config)
        self.time_film = TimeFiLM(**temporal_film_config)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        return_gradients: bool = False,
    ) -> SpatialDecoderManifest:
        """
        If return_gradients=True, returns (z_hat, DecoderGrads(dz_hat_dt)).
        """
        u, (B, T, H, W) = self._pre_film_features(x)  # u: [B,T,c_dec,H,W]
        gamma, beta = self._gamma_beta(t)  # [B,T,c_dec] each

        z_hat = self._decode_from_film(u, gamma, beta, B, T, H, W)

        if not return_gradients:
            return SpatialDecoderManifest(z_hat=z_hat)

        # Note: t.requires_grad is NOT needed for torch.func jacrev
        dgamma_dt, dbeta_dt = self._gamma_beta_time_grads(t)  # [B,T,c_dec] each
        dz_hat_dt = self._decode_dt_from_film(u, dgamma_dt, dbeta_dt, B, T, H, W)

        return SpatialDecoderManifest(z_hat=z_hat, dz_hat_dt=dz_hat_dt)

    def _pre_film_features(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        """
        Convert [B,T,C,H,W] -> u = conv(x) in [B,T,c_dec,H,W].
        """
        B, T, C, H, W = x.shape

        # Flatten time into batch for spatial ops
        x_bt = x.permute(0, 1, 3, 4, 2).reshape(B * T, H, W, C).permute(0, 3, 1, 2)  # [BT,C,H,W]

        if self.upsample:
            x_bt = F.interpolate(x_bt, mode="bilinear", size=(H, W), align_corners=False)

        u_bt = self.conv(x_bt)  # [BT,c_dec,H,W]
        c_dec = u_bt.shape[1]
        u = u_bt.view(B, T, c_dec, H, W)  # [B,T,c_dec,H,W]
        return u, (B, T, H, W)

    def _gamma_beta(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        t: [...], typically [B,T]
        -> gamma,beta: [..., c_dec]
        """
        emb = self.time_embedding(t)  # [..., E]
        gamma, beta = self.time_film(emb)  # [..., c_dec]
        return gamma, beta

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
        """
        Apply FiLM then final 1x1 conv and reshape to [B,7,L,T,H,W].
        """
        y = gamma[..., None, None] * u + beta[..., None, None]  # [B,T,c_dec,H,W]
        y_bt = y.view(B * T, y.shape[2], H, W)  # [BT,c_dec,H,W]
        out_bt = self.out(y_bt)  # [BT,out_channels,H,W]
        return self._reshape_output(out_bt, B, T, H, W)

    def _decode_dt_from_film(
        self,
        u: torch.Tensor,
        dgamma_dt: torch.Tensor,
        dbeta_dt: torch.Tensor,
        B: int,
        T: int,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """
        Per-pixel time derivative under the assumption du/dt = 0:
        d/dt (gamma*u + beta) = (dgamma/dt)*u + dbeta/dt
        Then push through final 1x1 conv.
        """
        dy_dt = dgamma_dt[..., None, None] * u + dbeta_dt[..., None, None]  # [B,T,c_dec,H,W]
        dy_bt = dy_dt.view(B * T, dy_dt.shape[2], H, W)  # [BT,c_dec,H,W]
        dout_bt = self.out(dy_bt)  # [BT,out_channels,H,W]
        return self._reshape_output(dout_bt, B, T, H, W)

    def _reshape_output(self, out_bt: torch.Tensor, B: int, T: int, H: int, W: int) -> torch.Tensor:
        """
        out_bt: [BT,out_channels,H,W] -> [B,7,L,T,H,W]
        Assumes out_channels == 7*L.
        """
        out = out_bt.view(B, T, 7, self.L, H, W).permute(0, 2, 3, 1, 4, 5).contiguous()
        return out

    def _gamma_beta_time_grads(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
        dgamma_dt, dbeta_dt: [B,T,c_dec]
        """
        B, T = t.shape
        t_bt = t.reshape(-1)  # [BT]

        def one_scalar(ts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # ts is a scalar tensor (shape [])
            emb = self.time_embedding(ts)  # [E]
            g, b = self.time_film(emb)  # [c_dec], [c_dec]
            return g, b

        dg_bt = vmap(lambda ts: jacrev(lambda x: one_scalar(x)[0])(ts))(t_bt)  # [BT, c_dec]
        db_bt = vmap(lambda ts: jacrev(lambda x: one_scalar(x)[1])(ts))(t_bt)  # [BT, c_dec]

        c_dec = dg_bt.shape[-1]
        return dg_bt.view(B, T, c_dec), db_bt.view(B, T, c_dec)


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
