from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, Mapping

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import wandb
from pytorch_lightning import LightningModule

from src.data.balloon import AcquisitionConstants
from src.models.blocks import HeinzleSignal, SpatialDecoderManifest
from src.utils.plotting import plot_latent_layers, plot_neural_bold_layers


@dataclass(frozen=True)
class CollocationBatch:
    """Index tensors describing collocation points.

    t: [1, 1, n_times, n_space]   — shared across batch and layers
    h: [B, 1, n_times, n_space]   — per-sample spatial points
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
        self.pred_buffer = []
        self.neural_buffer = []
        self.bold_buffer = []
        self.source_layer_buffer = []
        self.source_position_buffer = []
        self.true_z_hat_buffer = []

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
        source_position: torch.Tensor | None = None,
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

        true_bold = (
            self.normaliser.denormalize(bold_norm) if self.normaliser is not None else bold_norm
        )

        # --- diagnostics ---
        # if source_position is not None:
        #     src_h = source_position[0, 0].long()
        #     src_w = source_position[0, 1].long()
        #     p_src = pred_bold[0, :, :, src_h, src_w]
        #     t_src = true_bold[0, :, :, src_h, src_w]
        #     print(f"pred_bold src | min: {p_src.min().item():.4f} max: {p_src.max().item():.4f} std: {p_src.std().item():.4f} mean: {p_src.mean().item():.4f}")
        #     print(f"true_bold src | min: {t_src.min().item():.4f} max: {t_src.max().item():.4f} std: {t_src.std().item():.4f} mean: {t_src.mean().item():.4f}")
        # print(
        #     f"pred_v | min: {pred_v.min().item():.4f} max: {pred_v.max().item():.4f} mean: {pred_v.mean().item():.4f} std: {pred_v.std().item():.4f}"
        # )
        # print(
        #     f"pred_q | min: {pred_q.min().item():.4f} max: {pred_q.max().item():.4f} mean: {pred_q.mean().item():.4f} std: {pred_q.std().item():.4f}"
        # )
        # print(
        #     f"normaliser running_mean: {self.normaliser.running_mean.item():.6f} running_std: {self.normaliser.running_var.sqrt().item():.6f} count: {self.normaliser.running_count.item()} frozen: {self.normaliser.frozen}"
        # )
        # -------------------

        # Collocation loss
        pred_bold_at = self._gather_bold_at(pred_bold, collocation)
        true_bold_at = self._gather_bold_at(true_bold, collocation)
        L = pred_bold_at.shape[1]
        colloc_loss = torch.stack(
            [F.mse_loss(pred_bold_at[:, l], true_bold_at[:, l]) for l in range(L)]
        ).mean()

        # Source voxel loss — full T, all layers, per sample
        B = pred_bold.shape[0]
        b_idx = torch.arange(B, device=pred_bold.device)
        src_h = source_position[:, 0].long()
        src_w = source_position[:, 1].long()
        pred_bold_src = pred_bold[b_idx, :, :, src_h, src_w]  # [B, L, T]
        true_bold_src = true_bold[b_idx, :, :, src_h, src_w]  # [B, L, T]
        src_loss = torch.stack(
            [F.mse_loss(pred_bold_src[:, l], true_bold_src[:, l]) for l in range(L)]
        ).mean()

        return colloc_loss + self.hparams.loss_config.lambda_src * src_loss

    def _sanitise_states(self, states: dict[str, Any]) -> dict[str, Any]:
        for key, value in states.items():
            value = torch.nan_to_num(value, nan=0.0, posinf=1e3, neginf=-1e3)
            if key in ("f", "v", "q"):
                value = torch.clamp(value, min=1e-3)
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

        ds_dt = MICH._gather_grad_at(dz_hat_dt, layer, idx, signal="s")
        s_target = x - self.hparams.haemo.kappa * s - self.hparams.haemo.gamma * (f - 1)
        s_loss = F.mse_loss(ds_dt[:, burn_in:], s_target[:, burn_in:])

        df_dt = MICH._gather_grad_at(dz_hat_dt, layer, idx, signal="f")
        f_loss = F.mse_loss(df_dt[:, burn_in:], s[:, burn_in:])

        dv_dt = MICH._gather_grad_at(dz_hat_dt, layer, idx, signal="v")
        target_vdot = f - v ** (1 / self.hparams.haemo.alpha)
        if layer > 0:
            vstar_deeper = MICH._gather_z_hat_at(z_hat, idx, signal="vstar")[:, layer - 1]
            target_vdot += self.hparams.haemo.lambda_d * vstar_deeper
        v_loss = F.mse_loss(dv_dt[:, burn_in:], target_vdot[:, burn_in:] / self.hparams.haemo.tau)

        dq_dt = MICH._gather_grad_at(dz_hat_dt, layer, idx, signal="q")
        target_qdot = f * (
            1 - (1 - self.hparams.acquisition.E0) ** (1 / f)
        ) / self.hparams.acquisition.E0 - q * v ** (1 / self.hparams.haemo.alpha - 1)
        if layer > 0:
            qstar_deeper = MICH._gather_z_hat_at(z_hat, idx, signal="qstar")[:, layer - 1]
            target_qdot += self.hparams.haemo.lambda_d * qstar_deeper

        q_loss = F.mse_loss(dq_dt[:, burn_in:], target_qdot[:, burn_in:] / self.hparams.haemo.tau)

        dv_star_dt = MICH._gather_grad_at(dz_hat_dt, layer, idx, signal="vstar")
        v_star_target = (-v_star + v - 1) / self.hparams.haemo.tau
        v_star_loss = F.mse_loss(dv_star_dt[:, burn_in:], v_star_target[:, burn_in:])

        dq_star_dt = MICH._gather_grad_at(dz_hat_dt, layer, idx, signal="qstar")
        q_star_target = (-q_star + q - 1) / self.hparams.haemo.tau
        q_star_loss = F.mse_loss(dq_star_dt[:, burn_in:], q_star_target[:, burn_in:])

        return (s_loss + f_loss + v_loss + q_loss + v_star_loss + q_star_loss) / 6.0

    def _physics_loss(
        self,
        z_hat: torch.Tensor,
        dz_hat_dt: torch.Tensor,
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
        tot_physics_loss = torch.tensor(0.0, device=z_hat.device, dtype=torch.float32)
        for layer in range(z_hat.shape[2]):
            tot_physics_loss += (
                self._compute_physics_layer_loss(
                    z_hat, dz_hat_dt, idx, layer=layer, burn_in=self.hparams.loss_config.burn_in
                ).float()
                / z_hat.shape[2]
            )

        # Smoothness of gradients
        dz_dt_fd = z_hat[:, :, :, 1:] - z_hat[:, :, :, :-1]  # [B, S, L, T-1, H, W]
        smoothness_loss = dz_dt_fd.pow(2).mean()
        return tot_physics_loss + self.hparams.loss_config.lambda_smooth * smoothness_loss

    def _shared_step(self, batch, stage: Literal["train", "val"]) -> MICHManifest:
        bold, neural = batch["bold"], batch["neural"]
        source_position = batch["source_position"]

        bold_norm = self.normaliser(bold, source_position) if self.normaliser is not None else bold

        sd_manifest = self(
            bold_norm,
            self._make_time_grid(
                B=bold.shape[0], T=bold.shape[2], device=bold.device, dtype=bold.dtype
            ),
            return_gradients=True,
            normalise=False,
        )
        z_hat = sd_manifest.z_hat
        dz_hat_dt = sd_manifest.grads

        data_loss = self._data_loss(z_hat, bold_norm, source_position=source_position)
        physics_loss = self._physics_loss(z_hat, dz_hat_dt, source_position=source_position)
        total_loss = (
            self.hparams.loss_config.lambda_data * data_loss
            + self.hparams.loss_config.lambda_physics * physics_loss
        )

        on_step = stage == "train"
        on_epoch = stage == "val"
        self.log(
            f"{stage}/data_loss",
            data_loss,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            f"{stage}/physics_loss",
            physics_loss,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            f"{stage}/total_loss",
            total_loss,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=True,
            sync_dist=True,
        )

        # Log unsynced per-rank metrics to each rank's own wandb run.
        # Only log during training and throttle to match log_every_n_steps so
        # the W&B step axis on rank 1 stays in sync with rank 0's WandbLogger.
        if (
            stage == "train"
            and not self.trainer.is_global_zero
            and getattr(self, "_rank_run", None) is not None
            and self.global_step % self.trainer.log_every_n_steps == 0
        ):
            self._rank_run.log(
                {
                    f"{stage}/data_loss_step": data_loss.item(),
                    f"{stage}/physics_loss_step": physics_loss.item(),
                    f"{stage}/total_loss_step": total_loss.item(),
                }
            )

        if stage == "train":
            return MICHManifest(
                data_loss=data_loss, physics_loss=physics_loss, total_loss=total_loss
            )
        elif stage == "val":
            return MICHManifest(
                data_loss=data_loss,
                physics_loss=physics_loss,
                total_loss=total_loss,
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

        # Each rank logs its own plots to its own W&B run — no gather needed.
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
        for i in range(pred_bold.shape[0]):
            image = plot_neural_bold_layers(
                pred_bold=pred_bold[i],
                true_bold=true_bold[i],
                pred_neural=pred_neural[i],
                true_neural=true_neural[i],
                source_layer=source_layer[i],
                source_pos=source_pos[i],
            )
            if run is not None:
                run.log({"predictions": wandb.Image(image)})
            plt.close(image)

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
            if run is not None:
                run.log({"latents": wandb.Image(image)})
            plt.close(image)

    def on_fit_start(self) -> None:
        # Rank 0's wandb run is owned by WandbLogger — nothing to do here.
        # Non-zero ranks initialize their own run so per-rank metrics are tracked.
        if self.trainer.is_global_zero:
            return
        logger = self.trainer.logger
        base_name = logger._wandb_init.get("name", logger.name).rsplit(": rank", 1)[0]
        init_kwargs = {
            **logger._wandb_init,
            "name": f"{base_name}: rank {self.global_rank}",
            "reinit": True,
        }
        self._rank_run = wandb.init(**init_kwargs)

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
