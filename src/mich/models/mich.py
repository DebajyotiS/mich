from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, Mapping

import torch
from pytorch_lightning import LightningModule

from mich.models.blocks import SpatialDecoderManifest
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


class MICH(LearnablePhysioMixin, MICHLossMixin, MICHLoggingMixin, LightningModule):
    # CollocationMixin comes in transitively via MICHLossMixin.
    def __init__(
        self,
        heinzle_net: partial,
        normaliser: partial,
        optimizer: partial,
        scheduler: Mapping,
        loss_config: Mapping,
        learnable_physio: Mapping[str, bool] | None = None,
        image_log_every_n_val_calls: int = 1,
        *args,
        **kwargs: Any,
    ):
        super().__init__()
        torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
        self.save_hyperparameters(logger=False, ignore=["heinzle_net", "normaliser"])
        self.heinzle_net = heinzle_net
        self.normaliser = normaliser
        self._val_call_count = 0
        lc = self.hparams.loss_config
        self._bold_loss_fn = self._make_loss_fn(getattr(lc, "bold_loss", None))
        self._ode_loss_fn = self._make_loss_fn(getattr(lc, "ode_loss", None))
        self._supervision_loss_fn = self._make_loss_fn(getattr(lc, "supervision_loss", None))
        self._dzdt_loss_fn = self._make_loss_fn(getattr(lc, "dzdt_loss", None))
        self._x_phase_loss_fn = self._make_loss_fn(getattr(lc, "x_phase_loss", None))
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
            (lambda_physics_eff > 0.0)
            or (stage == "val")
            or getattr(lc, "supervise_dzdt", False)
            or getattr(lc, "supervise_x_phase", False)
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

        lambda_source_activity_eff = self._get_scheduled_lambda(
            getattr(lc, "lambda_source_activity", 0.0),
            getattr(lc, "warmup_steps_source_activity", 0),
            getattr(lc, "delay_steps_source_activity", 0),
        )

        if lambda_source_activity_eff > 0.0:
            source_activity_eps = getattr(lc, "source_activity_epsilon", 0.01)
            source_activity_loss = self._source_activity_loss(
                z_hat, source_position, source_layer, num_sources, source_activity_eps
            )
            total_loss = total_loss + lambda_source_activity_eff * source_activity_loss
        else:
            source_activity_loss = None

        lambda_quiescence_consistency_eff = self._get_scheduled_lambda(
            getattr(lc, "lambda_quiescence_consistency", 0.0),
            getattr(lc, "warmup_steps_quiescence_consistency", 0),
            getattr(lc, "delay_steps_quiescence_consistency", 0),
        )

        if lambda_quiescence_consistency_eff > 0.0:
            tau_s = getattr(lc, "quiescence_consistency_tau_s", 0.01)
            tau_f = getattr(lc, "quiescence_consistency_tau_f", 0.01)
            eps_x = getattr(lc, "quiescence_consistency_eps_x", 0.01)
            quiescence_consistency_loss = self._quiescence_consistency_loss(
                z_hat, tau_s, tau_f, eps_x
            )
            total_loss = (
                total_loss + lambda_quiescence_consistency_eff * quiescence_consistency_loss
            )
        else:
            quiescence_consistency_loss = None

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

        dzdt_supervision_loss = None
        lambda_dzdt_eff = 0.0
        per_sig_dzdt: dict = {}
        if batch.get("s") is not None and getattr(lc, "supervise_dzdt", False):
            lambda_dzdt_eff = self._get_scheduled_lambda(
                lc.lambda_dzdt_supervision,
                getattr(lc, "warmup_steps_dzdt_supervision", 0),
                getattr(lc, "delay_steps_dzdt_supervision", 0),
            )
            if lambda_dzdt_eff > 0:
                dzdt_supervision_loss, per_sig_dzdt = self._derivative_supervision_loss(
                    dz_hat_dt, batch, source_position, num_sources
                )
                total_loss = total_loss + lambda_dzdt_eff * dzdt_supervision_loss

        x_phase_loss = None
        lambda_x_phase_eff = 0.0
        xp_npearson_eff = None
        xp_pearson_eff = None
        if getattr(lc, "supervise_x_phase", False):
            xp_cfg = getattr(lc, "x_phase_loss", None)
            anneal_start_step = getattr(lc, "x_phase_ratio_anneal_start_step", 0)
            anneal_end_step = getattr(lc, "x_phase_ratio_anneal_end_step", 0)
            lambda_x_phase_target = getattr(lc, "lambda_x_phase", 0.0)
            if anneal_end_step > anneal_start_step:
                lambda_x_phase_target = self._anneal_between(
                    getattr(lc, "lambda_x_phase_start", lambda_x_phase_target),
                    getattr(lc, "lambda_x_phase_end", lambda_x_phase_target),
                    anneal_start_step,
                    anneal_end_step,
                )
            lambda_x_phase_eff = self._get_scheduled_lambda(
                lambda_x_phase_target,
                getattr(lc, "warmup_steps_x_phase", 0),
                getattr(lc, "delay_steps_x_phase", 0),
            )
            if xp_cfg is not None and anneal_end_step > anneal_start_step:
                xp_cfg.lambda_npearson = self._anneal_between(
                    getattr(lc, "x_phase_npearson_start", xp_cfg.lambda_npearson),
                    getattr(lc, "x_phase_npearson_end", xp_cfg.lambda_npearson),
                    anneal_start_step,
                    anneal_end_step,
                )
                xp_cfg.lambda_pearson = self._anneal_between(
                    getattr(lc, "x_phase_pearson_start", xp_cfg.lambda_pearson),
                    getattr(lc, "x_phase_pearson_end", xp_cfg.lambda_pearson),
                    anneal_start_step,
                    anneal_end_step,
                )
            if xp_cfg is not None:
                xp_npearson_eff = xp_cfg.lambda_npearson
                xp_pearson_eff = xp_cfg.lambda_pearson
            if lambda_x_phase_eff > 0:
                x_phase_loss = self._x_phase_loss(z_hat, dz_hat_dt, source_position, num_sources)
                total_loss = total_loss + lambda_x_phase_eff * x_phase_loss

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
        if _colloc_loss is not None:
            self.log(
                f"{stage}/loss/collocation",
                _colloc_loss,
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
        if dzdt_supervision_loss is not None:
            self.log(
                f"{stage}/loss/dzdt_supervision",
                dzdt_supervision_loss,
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=False,
                sync_dist=True,
                logger=_to_logger,
            )
        if x_phase_loss is not None:
            self.log(
                f"{stage}/loss/x_phase",
                x_phase_loss,
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=False,
                sync_dist=True,
                logger=_to_logger,
            )
        if xp_npearson_eff is not None:
            self.log(
                f"{stage}/x_phase_loss/lambda_npearson",
                xp_npearson_eff,
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=False,
                sync_dist=True,
                logger=_to_logger,
            )
            self.log(
                f"{stage}/x_phase_loss/lambda_pearson",
                xp_pearson_eff,
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=False,
                sync_dist=True,
                logger=_to_logger,
            )
            self.log(
                f"{stage}/x_phase_loss/lambda_x_phase_eff",
                lambda_x_phase_eff,
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=False,
                sync_dist=True,
                logger=_to_logger,
            )

        _adapter = getattr(self, "_adapter", None)
        if stage == "train":
            self._pending_train_log = None
        if (
            stage == "train"
            and _adapter is not None
            and self.global_step % self.trainer.log_every_n_steps == 0
        ):
            log_dict = {
                "global_step": self.global_step,
                # top-level loss scalars
                "train/loss/total": total_loss.item(),
                "train/loss/data": data_loss.item(),
                "train/loss/physics": physics_loss.item(),
                "train/loss/collocation": _colloc_loss.item() if _colloc_loss is not None else None,
                # weighted contributions
                "train/loss_weighted/total": total_loss.item(),
                "train/loss_weighted/data": (data_loss * lc.lambda_data).item(),
                "train/loss_weighted/physics": (physics_loss * lambda_physics_eff).item(),
                "train/loss_weighted/src": (src_loss * lc.lambda_src).item(),
                # scheduled lambdas
                "parameters/lambda_physics": lambda_physics_eff,
                "parameters/lambda_smooth": lambda_smooth_eff,
                "parameters/lambda_source_activity": lambda_source_activity_eff,
                "parameters/lambda_quiescence_consistency": lambda_quiescence_consistency_eff,
                **(
                    {
                        "train/loss/source_activity": source_activity_loss.item(),
                        "train/loss_weighted/source_activity": (
                            source_activity_loss * lambda_source_activity_eff
                        ).item(),
                    }
                    if source_activity_loss is not None
                    else {}
                ),
                **(
                    {
                        "train/loss/quiescence_consistency": quiescence_consistency_loss.item(),
                        "train/loss_weighted/quiescence_consistency": (
                            quiescence_consistency_loss * lambda_quiescence_consistency_eff
                        ).item(),
                    }
                    if quiescence_consistency_loss is not None
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
            if dzdt_supervision_loss is not None:
                log_dict.update(
                    {
                        "train/loss/dzdt_supervision": dzdt_supervision_loss.item(),
                        "train/loss_weighted/dzdt_supervision": (
                            dzdt_supervision_loss * lambda_dzdt_eff
                        ).item(),
                        **{f"dzdt_supervision/src_{k}": v.item() for k, v in per_sig_dzdt.items()},
                    }
                )
            if x_phase_loss is not None:
                log_dict.update(
                    {
                        "train/loss/x_phase": x_phase_loss.item(),
                        "train/loss_weighted/x_phase": (x_phase_loss * lambda_x_phase_eff).item(),
                    }
                )
            self._pending_train_log = log_dict

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

    @staticmethod
    def _pick_off_source_voxel(
        occupied_h: torch.Tensor, occupied_w: torch.Tensor, H: int, W: int
    ) -> tuple[int, int]:
        """Uniformly sample an (h, w) grid position that is NOT any of the given
        occupied (source) positions -- a deliberate off-source voxel for validation
        plots, as opposed to the incidental off-source coverage that falls out of
        plotting every layer at a single source's column."""
        occupied = {
            (int(h), int(w)) for h, w in zip(occupied_h.tolist(), occupied_w.tolist(), strict=True)
        }
        candidates = [(hh, ww) for hh in range(H) for ww in range(W) if (hh, ww) not in occupied]
        if not candidates:
            raise ValueError(
                f"No off-source voxel available: all {H * W} grid positions are occupied "
                "by sources."
            )
        idx = torch.randint(0, len(candidates), (1,)).item()
        return candidates[idx]

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

        self._val_call_count += 1
        log_images = (
            self._val_call_count % max(1, getattr(self.hparams, "image_log_every_n_val_calls", 1))
            == 0
        )

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

        pred_neural_full = z_hat[:, self._signal_index("x")]  # [B, L, T, H, W]
        B_full, L_full, T, H_full, W_full = pred_neural_full.shape
        pred_flat = pred_neural_full.permute(0, 1, 3, 4, 2).reshape(-1, T)
        true_flat = neural.permute(0, 1, 3, 4, 2).reshape(-1, T)
        layer_ids = (
            torch.arange(L_full, device=pred_flat.device)
            .view(1, L_full, 1, 1)
            .expand(B_full, L_full, H_full, W_full)
            .reshape(-1)
        )
        max_rows = 20000
        if pred_flat.shape[0] > max_rows:
            grid_idx = torch.randperm(pred_flat.shape[0], device=pred_flat.device)[:max_rows]
            pred_flat, true_flat, layer_ids = (
                pred_flat[grid_idx],
                true_flat[grid_idx],
                layer_ids[grid_idx],
            )

        grid_metrics_raw = self._neural_recovery_metrics(pred_flat, true_flat)
        grid_metrics = {
            "val/neural/grid_pearson": grid_metrics_raw["val/neural/pearson"],
        }
        # Per-layer breakdown of the same sampled rows -- pooling across layers can hide
        # a single hallucinating layer behind others that look fine. Always emitted
        # (including L=1, where it trivially matches the pooled value above) so a
        # single-layer run's dashboards keep working unchanged.
        for layer_idx in range(L_full):
            layer_mask = layer_ids == layer_idx
            if layer_mask.any():
                layer_metrics_raw = self._neural_recovery_metrics(
                    pred_flat[layer_mask], true_flat[layer_mask]
                )
                grid_metrics[f"val/neural/grid_pearson_layer{layer_idx}"] = layer_metrics_raw[
                    "val/neural/pearson"
                ]

        adapter = getattr(self, "_adapter", None)
        if adapter is not None:
            # commit=False when images follow so this merges into the same wandb history
            # row as the media logged just below, instead of a separate row -- see the
            # _pending_train_log comment above for why that matters.
            adapter.log(
                {"global_step": self.global_step, **metrics, **grid_metrics},
                commit=not log_images,
            )
        for k, v in {**metrics, **grid_metrics}.items():
            self.log(k, v, on_epoch=True, sync_dist=True, logger=True)

        if not log_images:
            return

        # Each rank logs its own plots to its own run -- no gather needed.
        subset = min(10, bold.shape[0])
        random_indices = torch.randperm(bold.shape[0])[:subset]
        subset_bold = bold[random_indices]
        subset_neural = neural[random_indices]
        subset_true_z_hat = true_zhat[random_indices] if true_zhat is not None else None
        subset_z_hat = z_hat[random_indices]
        subset_src_pos = source_position[random_indices]  # [B, S, 2]
        subset_src_layer = source_layer[random_indices]  # [B, S]
        subset_num_sources = num_sources[random_indices]  # [B]

        # Split the plotted samples 5 source-voxel / 5 off-source-voxel (fewer off-source
        # if subset < 10) so validation media always shows both what the model does at real
        # activity and what it does on quiescent background -- the latter is where
        # hallucination (grid_pearson dropping despite good source recovery, see above)
        # would actually be visible, and was previously never plotted at all.
        n_source = min(5, subset - subset // 2)  # ceil(subset/2) capped at 5; 10 -> 5/5
        is_source_voxel = torch.zeros(subset, dtype=torch.bool)
        is_source_voxel[:n_source] = True

        # .clone() -- these are mutated in place below for the off-source samples, and
        # subset_src_pos (a view these come from) is still needed downstream, unmutated,
        # to label the samples' TRUE source positions/layers in the plots.
        subset_h = subset_src_pos[:, 0, 0].clone()
        subset_w = subset_src_pos[:, 0, 1].clone()
        H_grid, W_grid = bold.shape[-2], bold.shape[-1]
        for i in range(n_source, subset):
            n_valid = int(subset_num_sources[i])
            h_off, w_off = self._pick_off_source_voxel(
                subset_src_pos[i, :n_valid, 0], subset_src_pos[i, :n_valid, 1], H_grid, W_grid
            )
            subset_h[i] = h_off
            subset_w[i] = w_off

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
        voxel_pos = torch.stack([subset_h, subset_w], dim=1)  # [B, 2]

        self._plot_and_log_predictions(
            pred_bold=pred_bold,
            true_bold=subset_bold,
            pred_neural=pred_neural,
            true_neural=subset_neural,
            source_layer=subset_src_layer,
            source_pos=subset_src_pos,
            num_sources=subset_num_sources,
            voxel_pos=voxel_pos,
            is_source_voxel=is_source_voxel,
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
                voxel_pos=voxel_pos,
                is_source_voxel=is_source_voxel,
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
                voxel_pos=voxel_pos,
                is_source_voxel=is_source_voxel,
            )

    def configure_optimizers(self):
        optim = self.hparams.optimizer(self.parameters())
        sched = self.hparams.scheduler(optim)
        return {
            "optimizer": optim,
            "lr_scheduler": {"scheduler": sched, **self.hparams.lightning},
        }
