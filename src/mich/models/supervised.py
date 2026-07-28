"""Fully supervised baseline: BOLD → neural, no physics constraints."""

from __future__ import annotations

import types

import torch
from pytorch_lightning import LightningModule

from mich.models.blocks import FullySupervisedNet
from mich.models.mich_logging import MICHLoggingMixin
from mich.models.mich_losses import MICHLossMixin
from mich.utils.plotting import plot_neural_bold_layers
from mich.utils.run_adapters import make_rank_zero_adapter


class SupervisedMICH(LightningModule):
    """Encoder-only supervised baseline.

    Takes BOLD as input, predicts neural activity with a direct regression head.
    Loss: MSE + Pearson between predicted and true neural at the source voxel.
    No physics loss, no ODE constraints.

    Reuses `MICHLossMixin._make_loss_fn` (for the loss) and
    `MICHLoggingMixin._neural_recovery_metrics` (for validation metrics) as plain
    unbound static calls rather than duplicating their logic -- this class does
    not mix in either class, since it needs none of their other (collocation/
    physics-loss/per-rank-adapter) machinery.
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
        # Always MSE + Pearson (unlike MICHLossMixin's other callers, this model's
        # loss_config has no `type` field of its own to select a different combination).
        self._loss_fn = MICHLossMixin._make_loss_fn(
            types.SimpleNamespace(
                type="mse+pearson",
                lambda_pearson=getattr(loss_config, "lambda_pearson", 1.0),
            )
        )
        self._pred_buffer: list[torch.Tensor] = []
        self._neural_buffer: list[torch.Tensor] = []
        self._bold_buffer: list[torch.Tensor] = []
        self._src_pos_buffer: list[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Forward / shared step
    # ------------------------------------------------------------------

    def forward(self, bold: torch.Tensor) -> torch.Tensor:
        """[B, L, T, H, W] BOLD -> [B, L, T, H, W] predicted neural activity,
        normalising with `self.normaliser.normalize` (the running statistics,
        no update) if a normaliser is set."""
        bold_norm = self.normaliser.normalize(bold) if self.normaliser is not None else bold
        return self.net(bold_norm)  # [B, L, T, H, W]

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        """Forward pass, loss at the single source voxel only (this model has
        no per-layer source metadata, unlike `mich.MICH`: `batch["source_position"]`
        here is `[B, 2]`, not `[B, S, 2]`), and (for `stage="val"`) buffer this
        batch for `on_validation_epoch_end`.

        Args:
            batch: "bold", "neural": [B, L, T, H, W]; "source_position": [B, 2].
            stage: "train" or "val" -- controls `self.log`'s on_step/on_epoch/
                logger flags and whether this batch is buffered.

        Returns:
            Scalar loss (`self._loss` on the source-voxel slice).
        """
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

        loss = self._loss_fn(pred_src, true_src)

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
        """Aggregate the epoch's buffered validation batches, log recovery
        metrics (`_neural_recovery_metrics`) at the source voxel over the full
        val set, and log a `plot_neural_bold_layers` plot for up to 10 random
        samples -- BOLD input is plotted in both the "pred" and "true" BOLD
        slots (this model has no BOLD prediction of its own to show).

        Note:
            Unlike `mich.MICH`'s version, this uses `make_rank_zero_adapter`
            (rank-0-only, no per-rank DDP coordination) rather than the
            `on_fit_start`/`on_fit_end`-managed per-rank adapter -- this class
            implements neither of those hooks.
        """
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

        metrics = MICHLoggingMixin._neural_recovery_metrics(pred_src, neural_src)
        adapter = make_rank_zero_adapter(self.trainer)
        if adapter is not None:
            adapter.log({"global_step": self.global_step, **metrics})
        for k, v in metrics.items():
            self.log(k, v, on_epoch=True, sync_dist=True, logger=True)

        # Plot a few samples
        subset = min(10, N)
        idx = torch.randperm(N)[:subset]
        bold_src = bold[idx, :, :, src_h[idx], src_w[idx]].float()
        pred_plot = pred_src[idx].float()
        true_plot = neural_src[idx].float()

        if adapter is None:
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
            images.append(fig)
        adapter.log({"global_step": self.global_step, "val/predictions": images})
        import matplotlib.pyplot as plt

        for fig in images:
            plt.close(fig)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        optim = self.hparams.optimizer(self.parameters())
        sched = self.hparams.scheduler(optim)
        return {"optimizer": optim, "lr_scheduler": {"scheduler": sched, **self.hparams.lightning}}
