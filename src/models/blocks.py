from dataclasses import dataclass
from typing import Any, Literal, Mapping, overload

import rootutils
import torch
from torch import nn
from torch.func import jacrev, vmap
from torch.nn import functional as F

root = rootutils.setup_root(__file__, pythonpath=True)
from src.utils.torch_utils import get_activation

HeinzleSignal = Literal["x", "s", "f", "v", "q", "vstar", "qstar"]


@dataclass(frozen=True)
class SpatialDecoderManifest:
    z_hat: torch.Tensor  # [B, 7, L, T, H, W]
    dz_hat_dt: torch.Tensor | None = None


class MaskedLayerMixing(nn.Module):
    """
    Masked layer mixing (L -> L) with a 1x1 convolution, then expands to C channels.
    """

    def __init__(self, L: int = 3, C: int = 16, init_identity: bool = True):
        super().__init__()
        self.C = C
        self.L = L
        self.mask: torch.Tensor
        self._generate_mask()
        self._register_mask()

        self.W = nn.Parameter(torch.zeros((L, L, 1, 1)))  # [L, L, 1, 1]
        self.b = nn.Parameter(torch.zeros((L,)))  # [L]
        self.expand_net = nn.Conv2d(L, C, kernel_size=1, bias=True)
        if init_identity:
            with torch.no_grad():
                self.W.zero_()
                self.W[0, 0, 0, 0] = 1.0
                self.W[1, 1, 0, 0] = 1.0
                self.W[2, 2, 0, 0] = 1.0

    def _generate_mask(self):
        mask = torch.zeros((self.L, self.L))
        idx = torch.arange(self.L)
        mask[idx, idx] = 1.0
        mask[idx[1:], idx[:-1]] = 1.0
        self.register_buffer("mask", mask.view(self.L, self.L, 1, 1))  # [L, L, 1, 1]

    def _register_mask(self):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process input tensor through convolution and expansion layers.

        Args:
            x (torch.Tensor): Input tensor of shape [B, L, T, H, W] where:
                - B: batch size
                - L: number of layers (must match self.L)
                - T: temporal dimension
                - H: height
                - W: width

        Returns:
            torch.Tensor: Output tensor of shape [B, T, C, H, W] where:
                - B: batch size
                - T: temporal dimension
                - C: number of output channels (self.C)
                - H: height
                - W: width

        Raises:
            AssertionError: If input layer dimension L does not match self.L
        """
        B, L, T, H, W = x.shape
        assert L == self.L, f"Expected input with {self.L} layers, got {L}"

        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, L, H, W)  # [B*T, L, H, W]

        W_eff = self.W * self.mask
        if W_eff.dtype != x.dtype:
            W_eff = W_eff.to(x.dtype)
        if self.b.dtype != x.dtype:
            self.b = self.b.to(x.dtype)
        x = F.conv2d(x, W_eff, bias=self.b)  # [B*T, L, H, W]
        x = self.expand_net(x)  # [B*T, C, H, W]
        x = x.view(B, T, self.C, H, W)  # [B, T, C, H, W]
        return x


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
        self.num_freqs = num_freqs
        self.max_freq = max_freq

    def forward(self, t):
        # t: [B, T]
        freqs = torch.logspace(
            0.0,
            torch.log10(torch.tensor(self.max_freq, device=t.device, dtype=t.dtype)),
            steps=self.num_freqs,
            device=t.device,
            dtype=t.dtype,
        )  # [F]

        ang = 2 * torch.pi * t[..., None] * freqs  # [B, T, F]
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # [B, T, 2F]
        return emb


class TimeFiLM(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, activation: str, c_dec: int):
        super().__init__()
        self.linear = nn.Linear(embed_dim, hidden_dim)
        self.activation = get_activation(activation)
        self.out = nn.Linear(hidden_dim, 2 * c_dec)  # output gamma and beta for FiLM modulation

    def forward(self, e_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, E = e_t.shape
        out = self.linear(e_t)  # [B, T, hidden_dim]
        out = self.activation(out)
        out = self.out(out)  # [B, T, 2*c_dec]
        gamma, beta = out.chunk(2, dim=-1)  # [B, T, c_dec], [B, T, c_dec]
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
        t: [B,T] -> gamma,beta: [B,T,c_dec]
        """
        emb = self.time_embedding(t)  # [B,T,E]
        gamma, beta = self.time_film(emb)  # [B,T,c_dec] each
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
        Compute diagonal time derivatives for each (b,t):
        dgamma[b,t,:] / dt[b,t]  and  dbeta[b,t,:] / dt[b,t]

        Returns:
        dgamma_dt, dbeta_dt: [B,T,c_dec]
        """
        B, T = t.shape
        t_bt = t.reshape(B * T)  # [BT]

        def one(t_scalar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # scalar -> (gamma_vec, beta_vec) both [c_dec]
            tt = t_scalar.view(1, 1)
            g, b = self._gamma_beta(tt)
            return g.view(-1), b.view(-1)

        # elementwise derivatives wrt scalar input
        dg_bt = vmap(lambda ti: jacrev(lambda x: one(x)[0])(ti))(t_bt)  # [BT,c_dec]
        db_bt = vmap(lambda ti: jacrev(lambda x: one(x)[1])(ti))(t_bt)  # [BT,c_dec]

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
