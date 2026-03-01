from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, Mapping

import matplotlib.pyplot as plt  # for closing figures after logging to wandb
import torch
import torch.nn.functional as F
import wandb
from pytorch_lightning import LightningModule

from src.data.balloon import AcquisitionConstants
from src.models.blocks import HeinzleSignal, SpatialDecoderManifest
from src.utils.plotting import plot_layers


@dataclass(frozen=True)
class CollocationBatch:
    """Index tensors describing collocation points."""

    t: torch.Tensor  # [N]
    h: torch.Tensor  # [N]
    w: torch.Tensor  # [N]


@dataclass(frozen=True)
class MICHManifest:
    data_loss: torch.Tensor
    physics_loss: torch.Tensor
    total_loss: torch.Tensor
    bold: torch.Tensor | None = None  # [B, L, T, H, W]
    neural: torch.Tensor | None = None  # [B, L, T, H, W]
    z_hat: torch.Tensor | None = None  # [B, 7, L, T, H, W]


class MICH(LightningModule):
    """
    LightningModule interface for the BOLD Inversion network.
    """

    def __init__(
        self,
        heinzle_net: partial,
        optimizer: partial,
        scheduler: Mapping,
        loss_config: Mapping,
        *args,
        **kwargs: Any,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["heinzle_net"])
        self.heinzle_net = heinzle_net
        self.pred_buffer = []
        self.neural_buffer = []
        self.bold_buffer = []
        self.source_layer_buffer = []
        self.source_position_buffer = []

    def forward(
        self, bold: torch.Tensor, time: torch.Tensor, *, return_gradients: bool = False
    ) -> SpatialDecoderManifest:
        return self.heinzle_net(bold, time, return_gradients=return_gradients)

    @staticmethod
    def _make_time_grid(B: int, T: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Returns t: [B, T] in [0,1].
        """
        t = torch.linspace(0.0, 1.0, T, device=device, dtype=dtype).view(1, T).expand(B, T)
        return t

    @staticmethod
    def _signal_index(signal: HeinzleSignal | int) -> int:
        mapping = {"x": 0, "s": 1, "f": 2, "v": 3, "q": 4, "vstar": 5, "qstar": 6}
        if isinstance(signal, int):
            if 0 <= signal < 7:
                return signal
            raise IndexError(f"signal index must be in [0,6], got {signal}")
        return mapping[signal]

    @staticmethod
    def _gather_z_hat_at(
        z_hat: torch.Tensor, idx: CollocationBatch, *, signal: HeinzleSignal | int
    ) -> torch.Tensor:
        """
        z_hat: [B, 7, L, T, H, W]
        returns: [N]
        """
        s = MICH._signal_index(signal)
        b_idx = torch.arange(z_hat.shape[0], device=z_hat.device)[:, None, None, None, None]
        l_idx = torch.arange(z_hat.shape[2], device=z_hat.device)[None, :, None, None, None]
        return z_hat[b_idx, s, l_idx, idx.t, idx.h, idx.w]

    @staticmethod
    def _gather_neural_at(neural: torch.Tensor, idx: CollocationBatch) -> torch.Tensor:
        """
        neural: [B, L, T, H, W]
        returns: [N]
        """
        b_idx = torch.arange(neural.shape[0], device=neural.device)[:, None, None, None, None]
        l_idx = torch.arange(neural.shape[1], device=neural.device)[None, :, None, None, None]
        return neural[b_idx, l_idx, idx.t, idx.h, idx.w]

    @staticmethod
    def _gather_bold_at(bold: torch.Tensor, idx: CollocationBatch) -> torch.Tensor:
        """
        bold: [B, L, T, H, W]
        returns: [N]
        """
        b_idx = torch.arange(bold.shape[0], device=bold.device)[:, None, None, None, None]
        l_idx = torch.arange(bold.shape[1], device=bold.device)[None, :, None, None, None]
        return bold[b_idx, l_idx, idx.t, idx.h, idx.w]

    @staticmethod
    def _compute_bold(
        v: torch.Tensor, q: torch.Tensor, acquisition: AcquisitionConstants, V0: float
    ) -> torch.Tensor:
        k1 = acquisition.k1
        k2 = acquisition.k2
        k3 = acquisition.k3
        bold = V0 * (k1 * (1 - q) + k2 * (1 - q / v) + k3 * (1 - v))
        return bold

    @staticmethod
    def _compute_bold_at(
        z_hat: torch.Tensor, idx: CollocationBatch, acquisition: AcquisitionConstants, V0: float
    ) -> torch.Tensor:
        v = MICH._gather_z_hat_at(z_hat, idx, signal="v")
        q = MICH._gather_z_hat_at(z_hat, idx, signal="q")
        bold = MICH._compute_bold(v, q, acquisition, V0)
        return bold

    @staticmethod
    def _gather_grad_at(
        dz_hat_dt: torch.Tensor, layer: int, idx: CollocationBatch, *, signal: HeinzleSignal | int
    ) -> torch.Tensor:
        """
        dz_hat_dt: [B, 7, L, T, H, W]
        returns: [N]
        """
        s = MICH._signal_index(signal)
        b_idx = torch.arange(dz_hat_dt.shape[0], device=dz_hat_dt.device)[
            :, None, None, None, None, None
        ]
        # l_idx is the layer index, which we want to gather separately since it's not part of the collocation batch
        l_idx = torch.tensor(layer, device=dz_hat_dt.device)[None, None, None, None, None, None]
        return dz_hat_dt[b_idx, s, l_idx, idx.t, idx.h, idx.w]

    @staticmethod
    def _sample_collocation_indices(
        *,
        T: int,
        H: int,
        W: int,
        n_times: int,
        n_space: int,
        device: torch.device,
    ) -> CollocationBatch:
        """
        Sample collocation points uniformly at random.

        Returns tensors shaped:
            t: [1, 1, n_times, n_space]
            h: [1, 1, n_times, n_space]
            w: [1, 1, n_times, n_space]
        """

        # Sample independent time indices
        t = torch.randint(0, T, (n_times, n_space), device=device)

        # Sample independent spatial coordinates
        h = torch.randint(0, H, (n_times, n_space), device=device)
        w = torch.randint(0, W, (n_times, n_space), device=device)

        # Add leading dims to match your expected CollocationBatch format
        t = t.unsqueeze(0).unsqueeze(0)
        h = h.unsqueeze(0).unsqueeze(0)
        w = w.unsqueeze(0).unsqueeze(0)

        return CollocationBatch(t=t, h=h, w=w)

    def _data_loss(self, z_hat: torch.Tensor, bold: torch.Tensor) -> torch.Tensor:
        collocation = MICH._sample_collocation_indices(
            T=bold.shape[2],
            H=bold.shape[3],
            W=bold.shape[4],
            n_times=self.hparams.loss_config.n_time,
            n_space=self.hparams.loss_config.n_space,
            device=z_hat.device,
        )
        pred_bold = MICH._compute_bold_at(
            z_hat, collocation, acquisition=self.hparams.acquisition, V0=self.hparams.V0
        )
        true_bold = MICH._gather_bold_at(bold, collocation)
        data_loss = F.mse_loss(pred_bold, true_bold)
        return data_loss

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
    ) -> torch.Tensor:
        # gather the relevant signals and their gradients at the collocation points for this layer
        x = MICH._gather_z_hat_at(z_hat, idx, signal="x")[:, layer]
        s = MICH._gather_z_hat_at(z_hat, idx, signal="s")[:, layer]
        f = MICH._gather_z_hat_at(z_hat, idx, signal="f")[:, layer]
        v = MICH._gather_z_hat_at(z_hat, idx, signal="v")[:, layer]
        q = MICH._gather_z_hat_at(z_hat, idx, signal="q")[:, layer]
        v_star = MICH._gather_z_hat_at(z_hat, idx, signal="vstar")[:, layer]
        q_star = MICH._gather_z_hat_at(z_hat, idx, signal="qstar")[:, layer]

        # sanitise the states
        states = self._sanitise_states(
            {
                "x": x,
                "s": s,
                "f": f,
                "v": v,
                "q": q,
                "vstar": v_star,
                "qstar": q_star,
            }
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

        ds_dt = MICH._gather_grad_at(dz_hat_dt, layer, idx, signal="s").squeeze(1).squeeze(1)
        target_sdot = x - self.hparams.haemo.kappa * s - self.hparams.haemo.gamma * (f - 1)
        s_loss = F.mse_loss(ds_dt, target_sdot)

        df_dt = MICH._gather_grad_at(dz_hat_dt, layer, idx, signal="f").squeeze(1).squeeze(1)
        target_fdot = s
        f_loss = F.mse_loss(df_dt, target_fdot)

        dv_dt = MICH._gather_grad_at(dz_hat_dt, layer, idx, signal="v").squeeze(1).squeeze(1)
        target_vdot = f - v ** (1 / self.hparams.haemo.alpha)
        if layer < z_hat.shape[2] - 1:
            v_star_next = MICH._gather_z_hat_at(z_hat, idx, signal="vstar")[:, layer + 1]
            target_vdot += self.hparams.haemo.lambda_d * v_star_next
        target_vdot = target_vdot / self.hparams.haemo.tau
        v_loss = F.mse_loss(dv_dt, target_vdot)

        dq_dt = MICH._gather_grad_at(dz_hat_dt, layer, idx, signal="q").squeeze(1).squeeze(1)
        target_qdot = f * (
            1 - (1 - self.hparams.acquisition.E0) ** (1 / f)
        ) / self.hparams.acquisition.E0 - q * v ** (1 / self.hparams.haemo.alpha - 1)
        if layer < z_hat.shape[2] - 1:
            q_star_next = MICH._gather_z_hat_at(z_hat, idx, signal="qstar")[:, layer + 1]
            target_qdot += self.hparams.haemo.lambda_d * q_star_next
        target_qdot = target_qdot / self.hparams.haemo.tau
        q_loss = F.mse_loss(dq_dt, target_qdot)

        dv_star_dt = (
            MICH._gather_grad_at(dz_hat_dt, layer, idx, signal="vstar").squeeze(1).squeeze(1)
        )
        target_v_star_dot = -v_star + v - 1
        target_v_star_dot = target_v_star_dot / self.hparams.haemo.tau
        v_star_loss = F.mse_loss(dv_star_dt, target_v_star_dot)

        dq_star_dt = (
            MICH._gather_grad_at(dz_hat_dt, layer, idx, signal="qstar").squeeze(1).squeeze(1)
        )
        target_q_star_dot = -q_star + q - 1
        target_q_star_dot = target_q_star_dot / self.hparams.haemo.tau
        q_star_loss = F.mse_loss(dq_star_dt, target_q_star_dot)

        layer_loss = s_loss + f_loss + v_loss + q_loss + v_star_loss + q_star_loss
        return layer_loss

    def _physics_loss(
        self,
        z_hat: torch.Tensor,
        dz_hat_dt: torch.Tensor,
    ) -> torch.Tensor:
        idx = MICH._sample_collocation_indices(
            T=z_hat.shape[3],
            H=z_hat.shape[4],
            W=z_hat.shape[5],
            n_times=self.hparams.loss_config.n_time,
            n_space=self.hparams.loss_config.n_space,
            device=z_hat.device,
        )
        tot_physics_loss = torch.tensor(0.0, device=z_hat.device)
        for layer in range(z_hat.shape[2]):
            layer_loss = self._compute_physics_layer_loss(z_hat, dz_hat_dt, idx, layer=layer)
            tot_physics_loss += layer_loss
        return tot_physics_loss

    def _shared_step(self, batch, stage: Literal["train", "val"]) -> MICHManifest:
        bold, neural = batch["bold"], batch["neural"]

        manifest = self(
            bold,
            self._make_time_grid(
                B=bold.shape[0], T=bold.shape[2], device=bold.device, dtype=bold.dtype
            ),
            return_gradients=True,
        )
        z_hat = manifest.z_hat
        dz_hat_dt = manifest.dz_hat_dt
        data_loss = self._data_loss(z_hat, bold)
        physics_loss = self._physics_loss(z_hat, dz_hat_dt)
        total_loss = (
            self.hparams.loss_config.lambda_data * data_loss
            + self.hparams.loss_config.lambda_physics * physics_loss
        )

        # log physics, data, and total losses separately
        self.log(f"{stage}/data_loss", data_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{stage}/physics_loss", physics_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{stage}/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)

        if stage == "train":
            return MICHManifest(
                data_loss=data_loss,
                physics_loss=physics_loss,
                total_loss=total_loss,
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
        manifest = self._shared_step(batch, stage="train")
        return manifest.total_loss

    def validation_step(self, batch, batch_idx):
        _num_pulses, _source_layer, source_position = (
            batch["num_pulses"],
            batch["source_layer"],
            batch["source_position"],
        )
        manifest = self._shared_step(batch, stage="val")

        z_hat = manifest.z_hat
        bold = manifest.bold
        neural = manifest.neural

        self.pred_buffer.append(z_hat)
        self.bold_buffer.append(bold)
        self.neural_buffer.append(neural)
        self.source_position_buffer.append(source_position)

        return manifest.total_loss

    def on_validation_epoch_end(self):
        # true bold are of shape B, L, T, H, W
        # true neural are of shape B, L, T, H, W
        # pred z_hat are of shape B, 7, L, T, H, W
        bold = torch.cat(self.bold_buffer, dim=0)
        neural = torch.cat(self.neural_buffer, dim=0)
        z_hat = torch.cat(self.pred_buffer, dim=0)
        source_position = torch.cat(self.source_position_buffer, dim=0)

        subset = max(10, bold.shape[0] // 10)  # log at least 10 samples, or 10% of the data
        random_indices = torch.randperm(bold.shape[0])[:subset]
        subset_bold = bold[random_indices]
        subset_neural = neural[random_indices]
        subset_z_hat = z_hat[random_indices]
        subset_source_pos = source_position[random_indices]

        subset_h = subset_source_pos[..., 0]
        subset_w = subset_source_pos[..., 1]
        batch_idx = torch.arange(subset_bold.shape[0])

        subset_bold = subset_bold[batch_idx, :, :, subset_h, subset_w]  # [subset, L,  T]
        subset_neural = subset_neural[batch_idx, :, :, subset_h, subset_w]  # [subset, L, T]
        subset_z_hat = subset_z_hat[batch_idx, :, :, :, subset_h, subset_w]  # [subset, 7, L, T]

        v_index = MICH._signal_index("v")
        q_index = MICH._signal_index("q")
        neural_index = MICH._signal_index("x")

        pred_bold = MICH._compute_bold(
            subset_z_hat[:, v_index],
            subset_z_hat[:, q_index],
            acquisition=self.hparams.acquisition,
            V0=self.hparams.V0,
        )  # [subset, L, T]
        pred_neural = subset_z_hat[:, neural_index]  # [subset, L, T]

        self._plot_and_log_predictions(
            pred_bold=pred_bold,
            true_bold=subset_bold,
            pred_neural=pred_neural,
            true_neural=subset_neural,
        )
        self.pred_buffer.clear()
        self.bold_buffer.clear()
        self.neural_buffer.clear()
        self.source_position_buffer.clear()

    def _plot_and_log_predictions(
        self,
        pred_bold: torch.Tensor,
        true_bold: torch.Tensor,
        pred_neural: torch.Tensor,
        true_neural: torch.Tensor,
    ):
        _num_samples = pred_bold.shape[0]
        for i in range(_num_samples):
            image = plot_layers(pred_bold[i], true_bold[i], pred_neural[i], true_neural[i])
            if wandb is not None:
                wandb.log({"predictions": wandb.Image(image)})
                # close the figure to free up memory
                plt.close(image)

    def configure_optimizers(self):
        optim = self.hparams.optimizer(self.parameters())
        sched = self.hparams.scheduler(optim)
        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": sched,
                **self.hparams.lightning,
            },
        }
