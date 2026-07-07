"""Fully supervised baseline: BOLD → neural, no physics constraints."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import wandb
from pytorch_lightning import LightningModule

from mich.models.blocks import FullySupervisedNet
from mich.utils.plotting import plot_neural_bold_layers


class SupervisedMICH(LightningModule):
    """Encoder-only supervised baseline.

    Takes BOLD as input, predicts neural activity with a direct regression head.
    Loss: MSE + Pearson between predicted and true neural at the source voxel.
    No physics loss, no ODE constraints.
    """

    def __init__(
        self,
        net: FullySupervisedNet,
        normaliser=None,
        loss_config=None,
        optimizer=None,
        scheduler=None,
        lightning=None,
        **kwargs,  # absorb scalar keys (L, C, c_enc, …) from yaml
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net", "normaliser"])
        self.net = net
        self.normaliser = normaliser
        self._pred_buffer: list[torch.Tensor] = []
        self._neural_buffer: list[torch.Tensor] = []
        self._bold_buffer: list[torch.Tensor] = []
        self._src_pos_buffer: list[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pearson_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        pred_c = pred - pred.mean(dim=-1, keepdim=True)
        true_c = true - true.mean(dim=-1, keepdim=True)
        num = (pred_c * true_c).sum(dim=-1)
        denom = (pred_c.norm(dim=-1) * true_c.norm(dim=-1)).clamp(min=1e-8)
        return (1.0 - num / denom).mean()

    def _loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        lc = self.hparams.loss_config
        lambda_pearson = getattr(lc, "lambda_pearson", 1.0)
        return F.mse_loss(pred, true) + lambda_pearson * self._pearson_loss(pred, true)

    # ------------------------------------------------------------------
    # Metrics (same as MICH)
    # ------------------------------------------------------------------

    @staticmethod
    def _neural_recovery_metrics(pred: torch.Tensor, true: torch.Tensor) -> dict[str, float]:
        pred, true = pred.float(), true.float()
        T = pred.shape[-1]
        flat_pred = pred.reshape(-1, T)
        flat_true = true.reshape(-1, T)

        ss_res = ((flat_true - flat_pred) ** 2).sum(dim=-1)
        ss_tot = ((flat_true - flat_true.mean(dim=-1, keepdim=True)) ** 2).sum(dim=-1)
        r2 = (1 - ss_res / ss_tot.clamp(min=1e-8)).mean().item()

        p_c = flat_pred - flat_pred.mean(dim=-1, keepdim=True)
        t_c = flat_true - flat_true.mean(dim=-1, keepdim=True)
        pearson = (
            ((p_c * t_c).sum(dim=-1) / (p_c.norm(dim=-1) * t_c.norm(dim=-1)).clamp(min=1e-8))
            .mean()
            .item()
        )

        xcorr = torch.fft.irfft(
            torch.fft.rfft(flat_true, n=2 * T) * torch.fft.rfft(flat_pred, n=2 * T).conj(),
            n=2 * T,
        )
        lags = torch.fft.fftfreq(2 * T, d=1.0 / (2 * T)).long().to(xcorr.device)
        peak_lag = lags[xcorr.argmax(dim=-1)].float().mean().item()

        return {
            "val/neural/r2": r2,
            "val/neural/pearson": pearson,
            "val/neural/lag_samples": peak_lag,
        }

    # ------------------------------------------------------------------
    # Forward / shared step
    # ------------------------------------------------------------------

    def forward(self, bold: torch.Tensor) -> torch.Tensor:
        bold_norm = self.normaliser.normalize(bold) if self.normaliser is not None else bold
        return self.net(bold_norm)  # [B, L, T, H, W]

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        bold = batch["bold"]
        true_neural = batch["neural"]
        source_position = batch["source_position"]

        pred_neural = self(bold)  # [B, L, T, H, W]

        B = pred_neural.shape[0]
        b_idx = torch.arange(B, device=pred_neural.device)
        src_h = source_position[:, 0].long()
        src_w = source_position[:, 1].long()

        pred_src = pred_neural[b_idx, :, :, src_h, src_w]  # [B, L, T]
        true_src = true_neural[b_idx, :, :, src_h, src_w]  # [B, L, T]

        loss = self._loss(pred_src, true_src)

        self.log(
            f"{stage}/loss/total",
            loss,
            on_step=(stage == "train"),
            on_epoch=(stage == "val"),
            prog_bar=True,
            sync_dist=True,
            logger=(stage == "val"),
        )

        if stage == "val":
            self._pred_buffer.append(pred_neural.detach().cpu())
            self._neural_buffer.append(true_neural.detach().cpu())
            self._bold_buffer.append(bold.detach().cpu())
            self._src_pos_buffer.append(source_position.detach().cpu())

        return loss

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, stage="val")

    def on_validation_epoch_end(self):
        if not self._pred_buffer:
            return

        pred = torch.cat(self._pred_buffer, dim=0)  # [N, L, T, H, W]
        neural = torch.cat(self._neural_buffer, dim=0)
        bold = torch.cat(self._bold_buffer, dim=0)
        src_pos = torch.cat(self._src_pos_buffer, dim=0)

        self._pred_buffer.clear()
        self._neural_buffer.clear()
        self._bold_buffer.clear()
        self._src_pos_buffer.clear()

        N = pred.shape[0]
        b_idx = torch.arange(N)
        src_h = src_pos[:, 0].long()
        src_w = src_pos[:, 1].long()

        pred_src = pred[b_idx, :, :, src_h, src_w]  # [N, L, T]
        neural_src = neural[b_idx, :, :, src_h, src_w]  # [N, L, T]

        metrics = self._neural_recovery_metrics(pred_src, neural_src)
        run = wandb.run
        if run is not None:
            run.log({"global_step": self.global_step, **metrics})
        for k, v in metrics.items():
            self.log(k, v, on_epoch=True, sync_dist=True, logger=True)

        # Plot a few samples
        subset = min(10, N)
        idx = torch.randperm(N)[:subset]
        bold_src = bold[idx, :, :, src_h[idx], src_w[idx]].float()
        pred_plot = pred_src[idx].float()
        true_plot = neural_src[idx].float()

        run = wandb.run
        if run is None:
            return
        images = []
        for i in range(subset):
            fig = plot_neural_bold_layers(
                pred_bold=bold_src[i],
                true_bold=bold_src[i],
                pred_neural=pred_plot[i],
                true_neural=true_plot[i],
                source_layer=torch.zeros(1, dtype=torch.long),
                source_pos=src_pos[idx[i] : idx[i] + 1],
            )
            images.append(wandb.Image(fig, caption=f"val sample {i}"))
            import matplotlib.pyplot as plt

            plt.close(fig)
        run.log({"global_step": self.global_step, "val/predictions": images})

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        optim = self.hparams.optimizer(self.parameters())
        sched = self.hparams.scheduler(optim)
        return {"optimizer": optim, "lr_scheduler": {"scheduler": sched, **self.hparams.lightning}}
