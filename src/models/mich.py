from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, Mapping

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import wandb
from pytorch_lightning import LightningModule

from src.data.balloon import AcquisitionConstants, PointSpreadFunction, _reflect_pad
from src.models.blocks import HeinzleSignal, SpatialDecoderManifest
from src.utils.plotting import plot_latent_layers, plot_neural_bold_layers


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


@dataclass(frozen=True)
class MICHManifest:
    data_loss: torch.Tensor
    physics_loss: torch.Tensor
    total_loss: torch.Tensor
    supervision_loss: torch.Tensor | None = None
    bold: torch.Tensor | None = None  # [B, L, T, H, W]
    neural: torch.Tensor | None = None  # [B, L, T, H, W]
    z_hat: torch.Tensor | None = None  # [B, 7, L, T, H, W]


class MICH(LightningModule):
    def __init__(
        self,
        heinzle_net: partial,
        normaliser: partial,
        optimizer: partial,
        scheduler: Mapping,
        loss_config: Mapping,
        *args,
        **kwargs: Any,
    ):
        super().__init__()
        torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
        self.save_hyperparameters(logger=False, ignore=["heinzle_net", "normaliser"])
        self.heinzle_net = heinzle_net
        self.normaliser = normaliser
        lc = self.hparams.loss_config
        self._bold_loss_fn = MICH._make_loss_fn(getattr(lc, "bold_loss", None))
        self._ode_loss_fn = MICH._make_loss_fn(getattr(lc, "ode_loss", None))
        self._supervision_loss_fn = MICH._make_loss_fn(getattr(lc, "supervision_loss", None))
        self.pred_buffer = []
        self.neural_buffer = []
        self.bold_buffer = []
        self.source_layer_buffer = []
        self.source_position_buffer = []
        self.true_z_hat_buffer = []

        # Build PSF objects and register 2D kernels as buffers so they move with the device.
        psf_fwhm = getattr(self.hparams, "psf_fwhm", None)
        if psf_fwhm is not None:
            self._psf = [PointSpreadFunction(fwhm=f) for f in psf_fwhm]
            for i, psf in enumerate(self._psf):
                self.register_buffer(
                    f"_psf_kernel_{i}",
                    torch.as_tensor(psf.kernel_2d(), dtype=torch.float32),
                )
        else:
            self._psf = None

    def forward(
        self,
        bold: torch.Tensor,
        time: torch.Tensor,
        *,
        return_gradients: bool = False,
        normalise: bool = False,
    ) -> SpatialDecoderManifest:
        if self.normaliser is not None and normalise:
            bold_norm = self.normaliser(bold)
        else:
            bold_norm = bold
        return self.heinzle_net(bold_norm, time, return_gradients=return_gradients)

    @staticmethod
    def _make_loss_fn(loss_cfg) -> callable:
        """Build a loss callable `(pred, true) -> scalar` from a loss config mapping.

        Supported types: mse, huber, pearson, mse+pearson, huber+pearson.
        Pearson correlation is computed over dim=1 (the time/sequence dimension).
        """
        if loss_cfg is None:
            return F.mse_loss

        loss_type = getattr(loss_cfg, "type", "mse")
        huber_delta = getattr(loss_cfg, "huber_delta", 1.0)
        lambda_pearson = getattr(loss_cfg, "lambda_pearson", 1.0)

        def _pearson(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
            pred_c = pred - pred.mean(dim=1, keepdim=True)
            true_c = true - true.mean(dim=1, keepdim=True)
            num = (pred_c * true_c).sum(dim=1)
            denom = (pred_c.norm(dim=1) * true_c.norm(dim=1)).clamp(min=1e-8)
            return (1.0 - num / denom).mean()

        if loss_type == "mse":
            return F.mse_loss
        elif loss_type == "huber":
            return partial(F.huber_loss, delta=huber_delta)
        elif loss_type == "pearson":
            return _pearson
        elif loss_type == "mse+pearson":

            def _mse_pearson(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
                lambda_npearson = getattr(loss_cfg, "lambda_npearson", 1.0)
                return lambda_npearson * F.mse_loss(pred, true) + lambda_pearson * _pearson(
                    pred, true
                )

            return _mse_pearson
        elif loss_type == "huber+pearson":

            def _huber_pearson(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
                lambda_npearson = getattr(loss_cfg, "lambda_npearson", 1.0)
                return lambda_npearson * F.huber_loss(
                    pred, true, delta=huber_delta
                ) + lambda_pearson * _pearson(pred, true)

            return _huber_pearson
        else:
            raise ValueError(
                f"Unrecognised loss type: {loss_type!r}. "
                "Must be one of: mse, huber, pearson, mse+pearson, huber+pearson"
            )

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
        s = torch.tensor(MICH._signal_index(signal), device=z_hat.device)
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
        s = torch.tensor(MICH._signal_index(signal), device=dz_hat_dt.device)
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
            B = source_position.shape[0]
            src_h = source_position[:, 0].long()
            src_w = source_position[:, 1].long()

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

            h_dense = (src_h[:, None, None] + off_h).clamp(0, H - 1)
            w_dense = (src_w[:, None, None] + off_w).clamp(0, W - 1)

            h_uniform = torch.randint(0, H, (B, n_times, n_uniform_s), device=device)
            w_uniform = torch.randint(0, W, (B, n_times, n_uniform_s), device=device)

            h = torch.cat([h_dense, h_uniform], dim=2).unsqueeze(1)
            w = torch.cat([w_dense, w_uniform], dim=2).unsqueeze(1)
        else:
            h = torch.randint(0, H, (1, 1, n_times, n_space), device=device)
            w = torch.randint(0, W, (1, 1, n_times, n_space), device=device)

        return CollocationBatch(t=t, h=h, w=w)

    @staticmethod
    def _compute_bold(
        v: torch.Tensor, q: torch.Tensor, acquisition: AcquisitionConstants, V0: float
    ) -> torch.Tensor:
        k1, k2, k3 = acquisition.k1, acquisition.k2, acquisition.k3
        return V0 * (k1 * (1 - q) + k2 * (1 - q / v) + k3 * (1 - v))

    @staticmethod
    def _compute_bold_at(
        z_hat: torch.Tensor, idx: CollocationBatch, acquisition: AcquisitionConstants, V0: float
    ) -> torch.Tensor:
        v = MICH._gather_z_hat_at(z_hat, idx, signal="v")
        q = MICH._gather_z_hat_at(z_hat, idx, signal="q")
        return MICH._compute_bold(v, q, acquisition, V0)

    def _data_loss(
        self,
        z_hat: torch.Tensor,
        bold_norm: torch.Tensor,
        source_position: torch.Tensor | None = None,
    ) -> torch.Tensor:
        collocation = MICH._sample_collocation_indices(
            T=bold_norm.shape[2],
            H=bold_norm.shape[3],
            W=bold_norm.shape[4],
            n_times=self.hparams.loss_config.n_time,
            n_space=self.hparams.loss_config.n_space,
            device=z_hat.device,
            source_position=source_position,
            dense_spatial_frac=self.hparams.loss_config.dense_spatial_frac,
            dense_spatial_radius=self.hparams.loss_config.dense_spatial_radius,
            dense_time_frac=self.hparams.loss_config.dense_time_frac,
            dense_time_lo=self.hparams.loss_config.dense_time_lo,
            dense_time_hi=self.hparams.loss_config.dense_time_hi,
            uniform_time_lo=self.hparams.loss_config.uniform_time_lo,
        )

        v_idx, q_idx = MICH._signal_index("v"), MICH._signal_index("q")
        pred_v = z_hat[:, v_idx]
        pred_q = z_hat[:, q_idx]
        pred_bold = MICH._compute_bold(
            pred_v, pred_q, acquisition=self.hparams.acquisition, V0=self.hparams.V0
        )

        # Apply per-layer Gaussian PSF to pred_bold [B, L, T, H, W]
        if self._psf is not None:
            B_size, L_size, T_size, H_size, W_size = pred_bold.shape
            layers_blurred = []
            for i in range(L_size):
                kernel = getattr(self, f"_psf_kernel_{i}")
                pad = kernel.shape[-1] // 2
                x = pred_bold[:, i].reshape(B_size * T_size, 1, H_size, W_size)
                x = _reflect_pad(_reflect_pad(x, pad, dim=2), pad, dim=3)
                layers_blurred.append(
                    F.conv2d(x, kernel, padding=0).reshape(B_size, T_size, H_size, W_size)
                )
            pred_bold = torch.stack(layers_blurred, dim=1)  # [B, L, T, H, W]

        true_bold = (
            self.normaliser.denormalize(bold_norm) if self.normaliser is not None else bold_norm
        )

        # Collocation loss -- per layer shape: [B, n_times, n_space]; Pearson over n_times (dim=1)
        pred_bold_at = self._gather_bold_at(pred_bold, collocation)
        true_bold_at = self._gather_bold_at(true_bold, collocation)
        L = pred_bold_at.shape[1]
        colloc_loss = torch.stack(
            [
                self._bold_loss_fn(pred_bold_at[:, layer], true_bold_at[:, layer])
                for layer in range(L)
            ]
        ).mean()

        # Source voxel loss -- full T, all layers, per sample; shape per layer: [B, T]; Pearson over T (dim=1)
        B = pred_bold.shape[0]
        b_idx = torch.arange(B, device=pred_bold.device)
        src_h = source_position[:, 0].long()
        src_w = source_position[:, 1].long()
        pred_bold_src = pred_bold[b_idx, :, :, src_h, src_w]  # [B, L, T]
        true_bold_src = true_bold[b_idx, :, :, src_h, src_w]  # [B, L, T]
        src_loss = torch.stack(
            [
                self._bold_loss_fn(pred_bold_src[:, layer], true_bold_src[:, layer])
                for layer in range(L)
            ]
        ).mean()

        total = colloc_loss + self.hparams.loss_config.lambda_src * src_loss
        return total, colloc_loss, src_loss

    def _sanitise_states(self, states: dict[str, Any]) -> dict[str, Any]:
        for key, value in states.items():
            value = torch.nan_to_num(value, nan=0.0, posinf=1e3, neginf=-1e3)
            if key in ("f", "v", "q"):
                value = torch.clamp(value, min=0.1)
            else:
                value = torch.clamp(value, min=-1e3, max=1e3)
            states[key] = value
        return states

    def _compute_physics_layer_loss(
        self,
        z_hat: torch.Tensor,
        dz_hat_dt: torch.Tensor,
        idx: CollocationBatch,
        layer: int,
        burn_in: int,
        order: str,
    ) -> torch.Tensor:
        x = MICH._gather_z_hat_at(z_hat, idx, signal="x")[:, layer]
        s = MICH._gather_z_hat_at(z_hat, idx, signal="s")[:, layer]
        f = MICH._gather_z_hat_at(z_hat, idx, signal="f")[:, layer]
        v = MICH._gather_z_hat_at(z_hat, idx, signal="v")[:, layer]
        q = MICH._gather_z_hat_at(z_hat, idx, signal="q")[:, layer]
        v_star = MICH._gather_z_hat_at(z_hat, idx, signal="vstar")[:, layer]
        q_star = MICH._gather_z_hat_at(z_hat, idx, signal="qstar")[:, layer]

        states = self._sanitise_states(
            {"x": x, "s": s, "f": f, "v": v, "q": q, "vstar": v_star, "qstar": q_star}
        )
        x, s, f, v, q, v_star, q_star = (
            states["x"],
            states["s"],
            states["f"],
            states["v"],
            states["q"],
            states["vstar"],
            states["qstar"],
        )

        s_scale = 1.0
        f_scale = 1.0
        v_scale = 1.0
        q_scale = 1.0
        v_star_scale = 1.0
        q_star_scale = 1.0

        alpha = self.hparams.haemo.alpha
        gamma = self.hparams.haemo.gamma
        kappa = self.hparams.haemo.kappa
        lambda_d = self.hparams.haemo.lambda_d
        tau = self.hparams.haemo.tau
        E0 = self.hparams.acquisition.E0

        ds_dt = MICH._gather_grad_at(dz_hat_dt, layer, idx, signal="s")
        s_target = x - kappa * s - gamma * (f - 1)
        s_loss = self._ode_loss_fn(ds_dt[:, burn_in:] / s_scale, s_target[:, burn_in:] / s_scale)

        df_dt = MICH._gather_grad_at(dz_hat_dt, layer, idx, signal="f")
        f_loss = self._ode_loss_fn(df_dt[:, burn_in:] / f_scale, s[:, burn_in:] / f_scale)

        dv_dt = MICH._gather_grad_at(dz_hat_dt, layer, idx, signal="v")
        if order == "exact":
            target_vdot = f - v ** (1 / alpha)
        elif order == "linear":
            f, v, q = f - 1, v - 1, q - 1
            target_vdot = f - v / alpha
            f, v, q = f + 1, v + 1, q + 1
        elif order == "quadratic":
            f, v, q = f - 1, v - 1, q - 1
            target_vdot = f - v / alpha - (1 - alpha) / (2 * alpha**2) * v**2
            f, v, q = f + 1, v + 1, q + 1
        if layer > 0:
            vstar_deeper = MICH._gather_z_hat_at(z_hat, idx, signal="vstar")[:, layer - 1]
            target_vdot += lambda_d * vstar_deeper
        v_loss = self._ode_loss_fn(
            dv_dt[:, burn_in:] / v_scale,
            (target_vdot[:, burn_in:] / tau) / v_scale,
        )

        dq_dt = MICH._gather_grad_at(dz_hat_dt, layer, idx, signal="q")
        if order == "exact":
            target_qdot = f * ( 1 - (1 - E0) ** (1 / f) ) / E0 - q * v ** (1 / alpha - 1)
        elif order == "linear":
            f, v, q = f - 1, v - 1, q - 1
            beta_1 = (1 - E0) * np.log(1 - E0) / E0
            target_qdot = (1 + beta_1) * f - q - (1/alpha - 1) * v
            f, v, q = f + 1, v + 1, q + 1
        elif order == "quadratic":
            f, v, q = f - 1, v - 1, q - 1
            beta_1 = (1 - E0) * np.log(1 - E0) / E0
            beta_2 = beta_1 * np.log(1 - E0) / 2
            target_qdot = ( (1 + beta_1) * f - q - (1/alpha - 1) * v
                - beta_2 * f**2
                - (1/alpha - 1) * v * q
                - (1/2) * (1/alpha - 1) * (1/alpha - 2) * v**2 )
            f, v, q = f + 1, v + 1, q + 1
        if layer > 0:
            qstar_deeper = MICH._gather_z_hat_at(z_hat, idx, signal="qstar")[:, layer - 1]
            target_qdot += lambda_d * qstar_deeper
        q_loss = self._ode_loss_fn(
            dq_dt[:, burn_in:] / q_scale,
            (target_qdot[:, burn_in:] / tau) / q_scale,
        )

        dv_star_dt = MICH._gather_grad_at(dz_hat_dt, layer, idx, signal="vstar")
        v_star_target = (-v_star + v - 1) / tau
        v_star_loss = self._ode_loss_fn(
            dv_star_dt[:, burn_in:] / v_star_scale, v_star_target[:, burn_in:] / v_star_scale
        )

        dq_star_dt = MICH._gather_grad_at(dz_hat_dt, layer, idx, signal="qstar")
        q_star_target = (-q_star + q - 1) / tau
        q_star_loss = self._ode_loss_fn(
            dq_star_dt[:, burn_in:] / q_star_scale, q_star_target[:, burn_in:] / q_star_scale
        )

        return {
            "s": s_loss,
            "f": f_loss,
            "v": v_loss,
            "q": q_loss,
            "vstar": v_star_loss,
            "qstar": q_star_loss,
        }

    # Mapping from z_hat signal name -> batch key
    _SUPERVISION_KEYS = (
        ("s", "s"),
        ("f", "f"),
        ("v", "v"),
        ("q", "q"),
        ("vstar", "v_star"),
        ("qstar", "q_star"),
    )

    def _supervision_loss(
        self,
        z_hat: torch.Tensor,  # [B, 7, L, T, H, W]
        batch: dict,
        source_position: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """MSE between predicted and ground-truth latent states at collocation points."""
        # Use T_min so collocation indices are valid for both z_hat and true latents.
        T_latent = batch["s"].shape[2]
        T_min = min(z_hat.shape[3], T_latent)
        lc = self.hparams.loss_config
        idx = MICH._sample_collocation_indices(
            T=T_min,
            H=z_hat.shape[4],
            W=z_hat.shape[5],
            n_times=lc.n_time,
            n_space=lc.n_space,
            device=z_hat.device,
            source_position=source_position,
            dense_spatial_frac=lc.dense_spatial_frac,
            dense_spatial_radius=lc.dense_spatial_radius,
            dense_time_frac=lc.dense_time_frac,
            dense_time_lo=lc.dense_time_lo,
            dense_time_hi=lc.dense_time_hi,
            uniform_time_lo=lc.uniform_time_lo,
        )
        per_sig: dict[str, torch.Tensor] = {}
        for sig, bk in self._SUPERVISION_KEYS:
            true = batch[bk].float()  # [B, L, T_latent, H, W]
            pred_at = MICH._gather_z_hat_at(z_hat, idx, signal=sig).float()  # [B, L, n_t, n_s]
            true_at = MICH._gather_bold_at(true, idx).float()  # [B, L, n_t, n_s]
            L = pred_at.shape[1]
            per_sig[sig] = torch.stack(
                [
                    self._supervision_loss_fn(pred_at[:, layer], true_at[:, layer])
                    for layer in range(L)
                ]
            ).mean()
        total = sum(per_sig.values()) / len(per_sig)
        return total, per_sig

    def _source_supervision_loss(
        self,
        z_hat: torch.Tensor,  # [B, 7, L, T, H, W]
        batch: dict,
        source_position: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """MSE between predicted and ground-truth latent states at the source voxel across all T."""
        B = z_hat.shape[0]
        b_idx = torch.arange(B, device=z_hat.device)
        src_h = source_position[:, 0].long()
        src_w = source_position[:, 1].long()
        T_latent = batch["s"].shape[2]
        T_pred = z_hat.shape[3]
        T_min = min(T_pred, T_latent)

        per_sig: dict[str, torch.Tensor] = {}
        for sig, bk in self._SUPERVISION_KEYS:
            true = batch[bk].float()  # [B, L, T_latent, H, W]
            pred = z_hat[:, MICH._signal_index(sig)].float()  # [B, L, T, H, W]
            pred_src = pred[b_idx, :, :T_min, src_h, src_w]  # [B, L, T_min]
            true_src = true[b_idx, :, :T_min, src_h, src_w]  # [B, L, T_min]
            L = pred_src.shape[1]
            per_sig[sig] = torch.stack(
                [
                    self._supervision_loss_fn(pred_src[:, layer_idx], true_src[:, layer_idx])
                    for layer_idx in range(L)
                ]
            ).mean()

        total = sum(per_sig.values()) / len(per_sig)
        return total, per_sig

    def _get_scheduled_lambda(
        self, lambda_target: float, warmup_steps: int, delay_steps: int = 0
    ) -> float:
        if warmup_steps <= 0 and delay_steps <= 0:
            return lambda_target
        if self.global_step < delay_steps:
            return 0.0
        ramp_step = self.global_step - delay_steps
        if warmup_steps <= 0:
            return lambda_target
        return min(1.0, ramp_step / warmup_steps) * lambda_target

    def _physics_loss( # add a parameter that says whether the loss is linear or nonlinear
        self,
        z_hat: torch.Tensor,
        dz_hat_dt: torch.Tensor,
        order: str,
        lambda_smooth: float = 0.0,
        source_position: torch.Tensor | None = None,
    ) -> torch.Tensor:
        idx = MICH._sample_collocation_indices(
            T=z_hat.shape[3],
            H=z_hat.shape[4],
            W=z_hat.shape[5],
            n_times=self.hparams.loss_config.n_time,
            n_space=self.hparams.loss_config.n_space,
            device=z_hat.device,
            source_position=source_position,
            dense_spatial_frac=self.hparams.loss_config.dense_spatial_frac,
            dense_spatial_radius=self.hparams.loss_config.dense_spatial_radius,
            dense_time_frac=self.hparams.loss_config.dense_time_frac,
            dense_time_lo=self.hparams.loss_config.dense_time_lo,
            dense_time_hi=self.hparams.loss_config.dense_time_hi,
            uniform_time_lo=self.hparams.loss_config.uniform_time_lo,
        )
        _eq_keys = ("s", "f", "v", "q", "vstar", "qstar")
        tot_physics_loss = torch.tensor(0.0, device=z_hat.device, dtype=torch.float32)
        per_eq = {k: torch.tensor(0.0, device=z_hat.device, dtype=torch.float32) for k in _eq_keys}
        n_layers = z_hat.shape[2]
        for layer in range(n_layers):
            layer_losses = self._compute_physics_layer_loss(
                z_hat, dz_hat_dt, idx, layer=layer, burn_in=self.hparams.loss_config.burn_in, order=order
            )
            layer_total = sum(layer_losses.values()).float() / 6.0
            tot_physics_loss = tot_physics_loss + layer_total / n_layers
            for k in _eq_keys:
                per_eq[k] = per_eq[k] + layer_losses[k].float() / n_layers

        # Smoothness of gradients
        dz_dt_fd = z_hat[:, :, :, 1:] - z_hat[:, :, :, :-1]  # [B, S, L, T-1, H, W]
        smoothness_loss = dz_dt_fd.pow(2).mean()
        return tot_physics_loss + lambda_smooth * smoothness_loss, per_eq

    def _shared_step(self, batch, stage: Literal["train", "val"]) -> MICHManifest:
        bold, neural = batch["bold"], batch["neural"]
        source_position = batch["source_position"]

        bold_norm = self.normaliser(bold, source_position) if self.normaliser is not None else bold

        lc = self.hparams.loss_config
        lambda_physics_eff = self._get_scheduled_lambda(
            lc.lambda_physics,
            getattr(lc, "warmup_steps_physics", 0),
            getattr(lc, "delay_steps_physics", 0),
        )
        lambda_smooth_eff = self._get_scheduled_lambda(
            lc.lambda_smooth,
            getattr(lc, "warmup_steps_smooth", 0),
            getattr(lc, "delay_steps_smooth", 0),
        )
        need_grads = (lambda_physics_eff > 0.0) or (stage == "val")
        sd_manifest = self(
            bold_norm,
            self._make_time_grid(
                B=bold.shape[0], T=bold.shape[2], device=bold.device, dtype=bold.dtype
            ),
            return_gradients=need_grads,
            normalise=False,
        )
        z_hat = sd_manifest.z_hat
        dz_hat_dt = sd_manifest.grads  # None when need_grads=False

        data_loss, colloc_loss, src_loss = self._data_loss(
            z_hat, bold_norm, source_position=source_position
        )
        if need_grads:
            physics_loss, per_eq_physics = self._physics_loss(
                z_hat, dz_hat_dt, lambda_smooth=lambda_smooth_eff, source_position=source_position, order=self.hparams.loss_config.order
            )
        else:
            physics_loss = torch.tensor(0.0, device=z_hat.device, dtype=torch.float32)
            per_eq_physics = {}
        total_loss = lc.lambda_data * data_loss + lambda_physics_eff * physics_loss

        supervision_loss = None
        lambda_supervision_eff = 0.0
        per_sig_supervision: dict = {}
        if batch.get("s") is not None and lc.lambda_supervision > 0:
            lambda_supervision_eff = self._get_scheduled_lambda(
                lc.lambda_supervision,
                getattr(lc, "warmup_steps_supervision", 0),
                getattr(lc, "delay_steps_supervision", 0),
            )
            supervision_loss, per_sig_supervision = self._source_supervision_loss(
                z_hat, batch, source_position
            )
            total_loss = total_loss + lambda_supervision_eff * supervision_loss

        # --- Lightning logger (progress bar + val checkpointing) ---
        # Train: logger=False —> W&B sink is direct run.log() below.
        # Val: logger=True —> ModelCheckpoint reads from PL logger.
        _to_logger = stage == "val"
        on_step = stage == "train"
        on_epoch = stage == "val"

        self.log(
            f"{stage}/loss/total",
            total_loss,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=True,
            sync_dist=True,
            logger=_to_logger,
        )
        self.log(
            f"{stage}/loss/data",
            data_loss,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=True,
            sync_dist=True,
            logger=_to_logger,
        )
        self.log(
            f"{stage}/loss/physics",
            physics_loss,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=False,
            sync_dist=True,
            logger=_to_logger,
        )
        if supervision_loss is not None:
            self.log(
                f"{stage}/loss/supervision",
                supervision_loss,
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=False,
                sync_dist=True,
                logger=_to_logger,
            )

        # --- Direct W&B logging (train only, throttled to log_every_n_steps) ---
        _direct_run = wandb.run if self.trainer.is_global_zero else getattr(self, "_rank_run", None)
        if (
            stage == "train"
            and _direct_run is not None
            and self.global_step % self.trainer.log_every_n_steps == 0
        ):
            log_dict = {
                "global_step": self.global_step,
                # top-level loss scalars
                "train/loss/total": total_loss.item(),
                "train/loss/data": data_loss.item(),
                "train/loss/physics": physics_loss.item(),
                # weighted contributions
                "train/loss_weighted/total": total_loss.item(),
                "train/loss_weighted/data": (data_loss * lc.lambda_data).item(),
                "train/loss_weighted/physics": (physics_loss * lambda_physics_eff).item(),
                "train/loss_weighted/src": (src_loss * lc.lambda_src).item(),
                # scheduled lambdas
                "parameters/lambda_physics": lambda_physics_eff,
                "parameters/lambda_smooth": lambda_smooth_eff,
                # ODE residuals — own section
                **{f"ode/{k}": v.item() for k, v in per_eq_physics.items()},
            }
            if supervision_loss is not None:
                log_dict.update(
                    {
                        "train/loss/supervision": supervision_loss.item(),
                        "train/loss_weighted/supervision": (
                            supervision_loss * lambda_supervision_eff
                        ).item(),
                        # latent supervision per signal — own section
                        **{
                            f"supervision/src_{k}": v.item() for k, v in per_sig_supervision.items()
                        },
                    }
                )
            _direct_run.log(log_dict)

        if stage == "train":
            return MICHManifest(
                data_loss=data_loss,
                physics_loss=physics_loss,
                total_loss=total_loss,
                supervision_loss=supervision_loss,
            )
        elif stage == "val":
            return MICHManifest(
                data_loss=data_loss,
                physics_loss=physics_loss,
                total_loss=total_loss,
                supervision_loss=supervision_loss,
                z_hat=z_hat,
                bold=bold,
                neural=neural,
            )
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        return self._shared_step(batch, stage="train").total_loss

    def validation_step(self, batch, batch_idx):
        _num_pulses, _source_layer, source_position = (
            batch["num_pulses"],
            batch["source_layer"],
            batch["source_position"],
        )

        manifest = self._shared_step(batch, stage="val")
        # True latents
        true_s = batch["s"]
        true_f = batch["f"]
        true_v = batch["v"]
        true_q = batch["q"]
        true_v_star = batch["v_star"]
        true_q_star = batch["q_star"]

        true_x = torch.empty_like(true_s)
        true_z_hat = torch.stack(
            [true_x, true_s, true_f, true_v, true_q, true_v_star, true_q_star], dim=1
        )
        if len(self.pred_buffer) < 100:
            self.pred_buffer.append(manifest.z_hat.detach().cpu())
            self.bold_buffer.append(manifest.bold.detach().cpu())
            self.neural_buffer.append(manifest.neural.detach().cpu())
            self.source_position_buffer.append(source_position.detach().cpu())
            self.source_layer_buffer.append(_source_layer.detach().cpu())
            self.true_z_hat_buffer.append(true_z_hat.detach().cpu())

        return manifest.total_loss

    def on_validation_epoch_end(self):
        bold = torch.cat(self.bold_buffer, dim=0)
        neural = torch.cat(self.neural_buffer, dim=0)
        z_hat = torch.cat(self.pred_buffer, dim=0)
        source_position = torch.cat(self.source_position_buffer, dim=0)
        source_layer = torch.cat(self.source_layer_buffer, dim=0)
        true_zhat = (
            torch.cat(self.true_z_hat_buffer, dim=0) if hasattr(self, "true_z_hat_buffer") else None
        )

        self.pred_buffer.clear()
        self.bold_buffer.clear()
        self.neural_buffer.clear()
        self.source_position_buffer.clear()
        self.source_layer_buffer.clear()
        self.true_z_hat_buffer.clear()

        # Each rank logs its own plots to its own W&B run -- no gather needed.
        subset = min(10, bold.shape[0])
        random_indices = torch.randperm(bold.shape[0])[:subset]
        subset_bold = bold[random_indices]
        subset_neural = neural[random_indices]
        subset_true_z_hat = true_zhat[random_indices] if true_zhat is not None else None
        subset_z_hat = z_hat[random_indices]
        subset_src_pos = source_position[random_indices]
        subset_src_layer = source_layer[random_indices]
        subset_h = subset_src_pos[..., 0]
        subset_w = subset_src_pos[..., 1]
        batch_idx = torch.arange(subset_bold.shape[0])

        subset_bold = subset_bold[batch_idx, :, :, subset_h, subset_w]
        subset_neural = subset_neural[batch_idx, :, :, subset_h, subset_w]
        subset_z_hat = subset_z_hat[batch_idx, :, :, :, subset_h, subset_w]
        subset_true_z_hat = subset_true_z_hat[batch_idx, :, :, :, subset_h, subset_w]

        pred_bold = MICH._compute_bold(
            subset_z_hat[:, MICH._signal_index("v")],
            subset_z_hat[:, MICH._signal_index("q")],
            acquisition=self.hparams.acquisition,
            V0=self.hparams.V0,
        )
        pred_neural = subset_z_hat[:, MICH._signal_index("x")]

        self._plot_and_log_predictions(
            pred_bold=pred_bold,
            true_bold=subset_bold,
            pred_neural=pred_neural,
            true_neural=subset_neural,
            source_layer=subset_src_layer,
            source_pos=subset_src_pos,
        )
        self._plot_and_log_latents(
            pred_s=subset_z_hat[:, MICH._signal_index("s")],
            true_s=subset_true_z_hat[:, MICH._signal_index("s")],
            pred_f=subset_z_hat[:, MICH._signal_index("f")],
            true_f=subset_true_z_hat[:, MICH._signal_index("f")],
            pred_v=subset_z_hat[:, MICH._signal_index("v")],
            true_v=subset_true_z_hat[:, MICH._signal_index("v")],
            pred_q=subset_z_hat[:, MICH._signal_index("q")],
            true_q=subset_true_z_hat[:, MICH._signal_index("q")],
            pred_v_star=subset_z_hat[:, MICH._signal_index("vstar")],
            true_v_star=subset_true_z_hat[:, MICH._signal_index("vstar")],
            pred_q_star=subset_z_hat[:, MICH._signal_index("qstar")],
            true_q_star=subset_true_z_hat[:, MICH._signal_index("qstar")],
        )

    def _plot_and_log_predictions(
        self, pred_bold, true_bold, pred_neural, true_neural, source_layer, source_pos
    ):
        run = getattr(self, "_rank_run", None) or wandb.run
        images = []
        for i in range(pred_bold.shape[0]):
            image = plot_neural_bold_layers(
                pred_bold=pred_bold[i],
                true_bold=true_bold[i],
                pred_neural=pred_neural[i],
                true_neural=true_neural[i],
                source_layer=source_layer[i],
                source_pos=source_pos[i],
            )
            images.append(wandb.Image(image))
            plt.close(image)
        if run is not None and images:
            run.log({"global_step": self.global_step, "media/predictions": images}, commit=False)

    def _plot_and_log_latents(
        self,
        pred_s,
        true_s,
        pred_f,
        true_f,
        pred_v,
        true_v,
        pred_q,
        true_q,
        pred_v_star,
        true_v_star,
        pred_q_star,
        true_q_star,
    ):
        run = getattr(self, "_rank_run", None) or wandb.run
        images = []
        for i in range(pred_s.shape[0]):
            image = plot_latent_layers(
                pred_f=pred_f[i],
                true_f=true_f[i],
                pred_s=pred_s[i],
                true_s=true_s[i],
                pred_v=pred_v[i],
                true_v=true_v[i],
                pred_q=pred_q[i],
                true_q=true_q[i],
                pred_v_star=pred_v_star[i],
                true_v_star=true_v_star[i],
                pred_q_star=pred_q_star[i],
                true_q_star=true_q_star[i],
                title="Latent States",
            )
            images.append(wandb.Image(image))
            plt.close(image)
        if run is not None and images:
            run.log({"global_step": self.global_step, "media/latents": images}, commit=True)

    def on_after_backward(self):
        if self.global_step == 0:
            return
        if self.global_step % self.trainer.log_every_n_steps == 0:
            _direct_run = (
                wandb.run if self.trainer.is_global_zero else getattr(self, "_rank_run", None)
            )
            if _direct_run is None:
                return

            log_dict = {"global_step": self.global_step}

            # FiLM linear vs output layer grad norms
            decoder = self.heinzle_net.spatial_decoder
            film = decoder.time_film
            linear_norms = [p.grad.norm() for p in film.linear.parameters() if p.grad is not None]
            out_norms = [p.grad.norm() for p in film.out.parameters() if p.grad is not None]
            if linear_norms:
                log_dict["gradients/film_linear_norm"] = torch.stack(linear_norms).norm().item()
            if out_norms:
                log_dict["gradients/film_out_norm"] = torch.stack(out_norms).norm().item()
            all_film_norms = linear_norms + out_norms
            if all_film_norms:
                log_dict["gradients/film_grad_norm"] = torch.stack(all_film_norms).norm().item()

            # Output head grad norms
            head_norms = [
                p.grad.norm()
                for head in decoder.out_heads
                for p in head.parameters()
                if p.grad is not None
            ]
            if head_norms:
                log_dict["gradients/out_heads_norm"] = torch.stack(head_norms).norm().item()

            _direct_run.log(log_dict)

    def on_fit_start(self) -> None:
        if self.trainer.is_global_zero:
            wandb.define_metric("global_step")
            wandb.define_metric("*", step_metric="global_step")
            return
        logger = self.trainer.logger
        base_name = logger._wandb_init.get("name", logger.name).rsplit(": rank", 1)[0]
        init_kwargs = {
            **logger._wandb_init,
            "name": f"{base_name}: rank {self.global_rank}",
            "reinit": True,
        }
        self._rank_run = wandb.init(**init_kwargs)
        self._rank_run.define_metric("global_step")
        self._rank_run.define_metric("*", step_metric="global_step")

    def on_fit_end(self) -> None:
        rank_run = getattr(self, "_rank_run", None)
        if not self.trainer.is_global_zero and rank_run is not None:
            rank_run.finish()
            self._rank_run = None

    def configure_optimizers(self):
        optim = self.hparams.optimizer(self.parameters())
        sched = self.hparams.scheduler(optim)
        return {
            "optimizer": optim,
            "lr_scheduler": {"scheduler": sched, **self.hparams.lightning},
        }
