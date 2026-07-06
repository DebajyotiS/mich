from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, Mapping

import torch
import wandb

from pytorch_lightning import LightningModule

from mich.models.blocks import SpatialDecoderManifest
from mich.models.collocation import CollocationMixin
from mich.models.mich_logging import MICHLoggingMixin
from mich.models.mich_losses import MICHLossMixin
from mich.models.physio import LearnablePhysioMixin


@dataclass(frozen=True)
class MICHManifest:
    data_loss: torch.Tensor
    physics_loss: torch.Tensor
    total_loss: torch.Tensor
    supervision_loss: torch.Tensor | None = None
    bold: torch.Tensor | None = None  # [B, L, T, H, W]
    neural: torch.Tensor | None = None  # [B, L, T, H, W]
    z_hat: torch.Tensor | None = None  # [B, 7, L, T, H, W]


class MICH(
    CollocationMixin, LearnablePhysioMixin, MICHLossMixin, MICHLoggingMixin, LightningModule
):
    def __init__(
        self,
        heinzle_net: partial,
        normaliser: partial,
        optimizer: partial,
        scheduler: Mapping,
        loss_config: Mapping,
        learnable_physio: Mapping[str, bool] | None = None,
        *args,
        **kwargs: Any,
    ):
        super().__init__()
        torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
        self.save_hyperparameters(logger=False, ignore=["heinzle_net", "normaliser"])
        self.heinzle_net = heinzle_net
        self.normaliser = normaliser
        lc = self.hparams.loss_config
        self._bold_loss_fn = self._make_loss_fn(getattr(lc, "bold_loss", None))
        self._ode_loss_fn = self._make_loss_fn(getattr(lc, "ode_loss", None))
        self._supervision_loss_fn = self._make_loss_fn(getattr(lc, "supervision_loss", None))
        self._dsdt_loss_fn = self._make_loss_fn(getattr(lc, "dsdt_loss", None))
        self.pred_buffer = []
        self.neural_buffer = []
        self.bold_buffer = []
        self.source_layer_buffer = []
        self.source_position_buffer = []
        self.num_sources_buffer = []
        self.true_z_hat_buffer = []

        self._setup_learnable_physio(learnable_physio)
        self._setup_psf()

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

    def _shared_step(self, batch, stage: Literal["train", "val"]) -> MICHManifest:
        bold, neural = batch["bold"], batch["neural"]
        source_position = batch["source_position"]  # [B, S, 2]
        source_layer = batch["source_layer"]  # [B, S]
        num_sources = batch["num_sources"]  # [B]

        bold_norm = (
            self.normaliser(bold, source_position, num_sources)
            if self.normaliser is not None
            else bold
        )

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
        need_grads = (
            (lambda_physics_eff > 0.0) or (stage == "val") or getattr(lc, "supervise_dsdt", False)
        )
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

        data_loss, _colloc_loss, src_loss = self._data_loss(
            z_hat, bold_norm, source_position=source_position, num_sources=num_sources
        )
        if need_grads:
            physics_loss, per_eq_physics = self._physics_loss(
                z_hat,
                dz_hat_dt,
                lambda_smooth=lambda_smooth_eff,
                source_position=source_position,
                num_sources=num_sources,
                order=self.hparams.loss_config.order,
            )
        else:
            physics_loss = torch.tensor(0.0, device=z_hat.device, dtype=torch.float32)
            per_eq_physics = {}
        total_loss = lc.lambda_data * data_loss + lambda_physics_eff * physics_loss

        lambda_antisteady_eff = self._get_scheduled_lambda(
            getattr(lc, "lambda_antisteady", 0.0),
            getattr(lc, "warmup_steps_antisteady", 0),
            getattr(lc, "delay_steps_antisteady", 0),
        )

        if lambda_antisteady_eff > 0.0:
            antisteady_loss = self._antisteady_loss(
                z_hat, source_position, source_layer, num_sources
            )
            total_loss = total_loss + lambda_antisteady_eff * antisteady_loss
        else:
            antisteady_loss = None

        supervision_loss = None
        lambda_supervision_eff = 0.0
        per_sig_supervision: dict = {}
        if batch.get("s") is not None and lc.lambda_supervision > 0:
            lambda_supervision_eff = self._get_scheduled_lambda(
                lc.lambda_supervision,
                getattr(lc, "warmup_steps_supervision", 0),
                getattr(lc, "delay_steps_supervision", 0),
            )
            if lambda_supervision_eff > 0:
                supervision_loss, per_sig_supervision = self._source_supervision_loss(
                    z_hat, batch, source_position, num_sources
                )
                total_loss = total_loss + lambda_supervision_eff * supervision_loss

        dsdt_supervision_loss = None
        lambda_dsdt_eff = 0.0
        per_sig_dsdt: dict = {}
        if batch.get("s") is not None and getattr(lc, "supervise_dsdt", False):
            lambda_dsdt_eff = self._get_scheduled_lambda(
                lc.lambda_dsdt_supervision,
                getattr(lc, "warmup_steps_dsdt_supervision", 0),
                getattr(lc, "delay_steps_dsdt_supervision", 0),
            )
            if lambda_dsdt_eff > 0:
                dsdt_supervision_loss, per_sig_dsdt = self._derivative_supervision_loss(
                    dz_hat_dt, batch, source_position, num_sources
                )
                total_loss = total_loss + lambda_dsdt_eff * dsdt_supervision_loss

        # Lightning logger (progress bar + val checkpointing)
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
        if dsdt_supervision_loss is not None:
            self.log(
                f"{stage}/loss/dzdt_supervision",
                dsdt_supervision_loss,
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=False,
                sync_dist=True,
                logger=_to_logger,
            )

        # Direct W&B logging (train only, throttled to log_every_n_steps)
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
                "parameters/lambda_antisteady": lambda_antisteady_eff,
                **(
                    {
                        "train/loss/antisteady": antisteady_loss.item(),
                        "train/loss_weighted/antisteady": (
                            antisteady_loss * lambda_antisteady_eff
                        ).item(),
                    }
                    if antisteady_loss is not None
                    else {}
                ),
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
            if dsdt_supervision_loss is not None:
                log_dict.update(
                    {
                        "train/loss/dzdt_supervision": dsdt_supervision_loss.item(),
                        "train/loss_weighted/dzdt_supervision": (
                            dsdt_supervision_loss * lambda_dsdt_eff
                        ).item(),
                        **{f"dzdt_supervision/src_{k}": v.item() for k, v in per_sig_dsdt.items()},
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
        source_layer, source_position, num_sources = (
            batch["source_layer"],
            batch["source_position"],
            batch["num_sources"],
        )

        manifest = self._shared_step(batch, stage="val")
        # True latents
        true_s = batch["s"]
        true_f = batch["f"]
        true_v = batch["v"]
        true_q = batch["q"]
        true_x = torch.empty_like(true_s)
        if "v_star" in batch and "q_star" in batch:
            true_z_hat = torch.stack(
                [true_x, true_s, true_f, true_v, true_q, batch["v_star"], batch["q_star"]], dim=1
            )
        else:
            true_z_hat = torch.stack([true_x, true_s, true_f, true_v, true_q], dim=1)
        if len(self.pred_buffer) < 100:
            self.pred_buffer.append(manifest.z_hat.detach().cpu())
            self.bold_buffer.append(manifest.bold.detach().cpu())
            self.neural_buffer.append(manifest.neural.detach().cpu())
            self.source_position_buffer.append(source_position.detach().cpu())
            self.source_layer_buffer.append(source_layer.detach().cpu())
            self.num_sources_buffer.append(num_sources.detach().cpu())
            self.true_z_hat_buffer.append(true_z_hat.detach().cpu())

        return manifest.total_loss

    def on_validation_epoch_end(self):
        bold = torch.cat(self.bold_buffer, dim=0)
        neural = torch.cat(self.neural_buffer, dim=0)
        z_hat = torch.cat(self.pred_buffer, dim=0)
        source_position = torch.cat(self.source_position_buffer, dim=0)
        source_layer = torch.cat(self.source_layer_buffer, dim=0)
        num_sources = torch.cat(self.num_sources_buffer, dim=0)
        true_zhat = (
            torch.cat(self.true_z_hat_buffer, dim=0) if hasattr(self, "true_z_hat_buffer") else None
        )

        self.pred_buffer.clear()
        self.bold_buffer.clear()
        self.neural_buffer.clear()
        self.source_position_buffer.clear()
        self.source_layer_buffer.clear()
        self.num_sources_buffer.clear()
        self.true_z_hat_buffer.clear()

        # Each rank logs its own plots to its own W&B run -- no gather needed.
        subset = min(10, bold.shape[0])
        random_indices = torch.randperm(bold.shape[0])[:subset]
        subset_bold = bold[random_indices]
        subset_neural = neural[random_indices]
        subset_true_z_hat = true_zhat[random_indices] if true_zhat is not None else None
        subset_z_hat = z_hat[random_indices]
        subset_src_pos = source_position[random_indices]  # [B, S, 2]
        subset_src_layer = source_layer[random_indices]  # [B, S]
        subset_num_sources = num_sources[random_indices]  # [B]
        # Trace-extraction below shows one representative voxel per sample (not per
        # source) to keep the existing single-voxel plot layout; slot 0 is always a
        # valid source since num_sources >= 1 for every sample.
        subset_h = subset_src_pos[:, 0, 0]
        subset_w = subset_src_pos[:, 0, 1]
        batch_idx = torch.arange(subset_bold.shape[0])

        # Compute pred_bold at full spatial resolution so PSF can be applied before indexing.
        pred_bold_full = self._compute_bold(
            subset_z_hat[:, self._signal_index("v")],
            subset_z_hat[:, self._signal_index("q")],
            acquisition=self._current_acquisition(),
            V0=self._physio("V0"),
        )  # [B, L, T, H, W]
        pred_bold_full = self._apply_psf_blur(pred_bold_full)

        subset_bold = subset_bold[batch_idx, :, :, subset_h, subset_w]
        subset_neural = subset_neural[batch_idx, :, :, subset_h, subset_w]
        subset_z_hat = subset_z_hat[batch_idx, :, :, :, subset_h, subset_w]
        subset_true_z_hat = subset_true_z_hat[batch_idx, :, :, :, subset_h, subset_w]

        pred_bold = pred_bold_full[batch_idx, :, :, subset_h, subset_w]  # [B, L, T]

        pred_neural = subset_z_hat[:, self._signal_index("x")]

        # Neural recovery metrics over the full val set (not just the plot subset),
        # averaged over every real source per sample (not just source slot 0) --
        # padded slots (index >= num_sources) are masked out before averaging.
        S = source_position.shape[1]
        all_src_h = source_position[..., 0].long()  # [B, S]
        all_src_w = source_position[..., 1].long()  # [B, S]
        all_batch = (
            torch.arange(z_hat.shape[0], device=z_hat.device).unsqueeze(1).expand(-1, S)
        )  # [B, S]
        all_pred_neural = z_hat[
            all_batch, self._signal_index("x"), :, :, all_src_h, all_src_w
        ]  # [B, S, L, T]
        all_true_neural = neural[all_batch, :, :, all_src_h, all_src_w]  # [B, S, L, T]
        src_mask = torch.arange(S, device=z_hat.device)[None, :] < num_sources[:, None]  # [B, S]
        metrics = self._neural_recovery_metrics(
            all_pred_neural[src_mask], all_true_neural[src_mask]
        )
        run = getattr(self, "_rank_run", None) or wandb.run
        if run is not None:
            run.log({"global_step": self.global_step, **metrics})
        for k, v in metrics.items():
            self.log(k, v, on_epoch=True, sync_dist=True, logger=True)

        self._plot_and_log_predictions(
            pred_bold=pred_bold,
            true_bold=subset_bold,
            pred_neural=pred_neural,
            true_neural=subset_neural,
            source_layer=subset_src_layer,
            source_pos=subset_src_pos,
            num_sources=subset_num_sources,
        )
        has_drain = subset_z_hat.shape[1] > 5
        if has_drain:
            self._plot_and_log_latents(
                pred_s=subset_z_hat[:, self._signal_index("s")],
                true_s=subset_true_z_hat[:, self._signal_index("s")],
                pred_f=subset_z_hat[:, self._signal_index("f")],
                true_f=subset_true_z_hat[:, self._signal_index("f")],
                pred_v=subset_z_hat[:, self._signal_index("v")],
                true_v=subset_true_z_hat[:, self._signal_index("v")],
                pred_q=subset_z_hat[:, self._signal_index("q")],
                true_q=subset_true_z_hat[:, self._signal_index("q")],
                pred_v_star=subset_z_hat[:, self._signal_index("vstar")],
                true_v_star=subset_true_z_hat[:, self._signal_index("vstar")],
                pred_q_star=subset_z_hat[:, self._signal_index("qstar")],
                true_q_star=subset_true_z_hat[:, self._signal_index("qstar")],
            )
        else:
            self._plot_and_log_latents(
                pred_s=subset_z_hat[:, self._signal_index("s")],
                true_s=subset_true_z_hat[:, self._signal_index("s")],
                pred_f=subset_z_hat[:, self._signal_index("f")],
                true_f=subset_true_z_hat[:, self._signal_index("f")],
                pred_v=subset_z_hat[:, self._signal_index("v")],
                true_v=subset_true_z_hat[:, self._signal_index("v")],
                pred_q=subset_z_hat[:, self._signal_index("q")],
                true_q=subset_true_z_hat[:, self._signal_index("q")],
            )

    def configure_optimizers(self):
        optim = self.hparams.optimizer(self.parameters())
        sched = self.hparams.scheduler(optim)
        return {
            "optimizer": optim,
            "lr_scheduler": {"scheduler": sched, **self.hparams.lightning},
        }
