"""Validation-time metrics, plotting, and rank-run/gradient-norm hooks for MICH."""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch
from pytorch_lightning.loggers import MLFlowLogger, WandbLogger

from mich.utils.plotting import plot_latent_layers, plot_neural_bold_layers
from mich.utils.run_adapters import make_run_adapter


class MICHLoggingMixin:
    """Neural-recovery metrics, plot assembly, and the PL hooks that manage per-rank
    runs and gradient-norm logging. Pure side effects -- no loss/physics math here.

    Reads/uses, none of which this class defines itself (mixed in alongside this
    class at the concrete model's `LightningModule` base, not inherited by it):
      - `self.global_step`, `self.trainer` (`LightningModule`); `self.log(...)`
        (`LightningModule`, via PyTorch Lightning).
      - `self.heinzle_net` (set in `mich.MICH.__init__`): `.spatial_decoder.time_film`,
        `.spatial_decoder.out_heads` -- read for their parameters' `.grad` in
        `on_after_backward`.
      - `self._pending_train_log` (dict or None): built by `mich.MICH._shared_step`
        each training step and consumed (then reset to None) by
        `on_after_backward` below -- the two must run in that order within a
        step, which Lightning's training loop guarantees (backward always
        follows the step that sets it).

    Also sets/owns `self._adapter` (a `RunAdapter` or None) via `on_fit_start`,
    read by every plotting/logging method below.
    """

    @staticmethod
    def _neural_recovery_metrics(
        pred: torch.Tensor,  # [B, L, T]
        true: torch.Tensor,  # [B, L, T]
    ) -> dict[str, float]:
        """R2, Pearson r, and peak cross-correlation lag, each averaged over every
        (batch, layer) pair.

        Peak lag is the argmax of the circular cross-correlation (computed via
        FFT, zero-padded to `2*T` to approximate a linear, non-wraparound
        correlation over the range it's evaluated at) between `true` and
        `pred`, in samples; positive means `pred` lags `true`.

        Args:
            pred, true: [B, L, T].

        Returns:
            {"val/neural/r2", "val/neural/pearson", "val/neural/lag_samples"}.

        Warning:
            `ss_tot`/pearson's denominator are `.clamp(min=1e-8)`, so a
            near-constant `true`/`pred` row doesn't raise but instead produces
            a very large-magnitude (not NaN/inf) R2 or an ~0 Pearson
            contribution for that row.
        """
        pred = pred.float()
        true = true.float()
        T = pred.shape[-1]
        flat_pred = pred.reshape(-1, T)  # [B*L, T]
        flat_true = true.reshape(-1, T)

        # R2
        ss_res = ((flat_true - flat_pred) ** 2).sum(dim=-1)
        ss_tot = ((flat_true - flat_true.mean(dim=-1, keepdim=True)) ** 2).sum(dim=-1)
        r2 = (1 - ss_res / ss_tot.clamp(min=1e-8)).mean().item()

        # Pearson
        p_c = flat_pred - flat_pred.mean(dim=-1, keepdim=True)
        t_c = flat_true - flat_true.mean(dim=-1, keepdim=True)
        pearson = (
            ((p_c * t_c).sum(dim=-1) / (p_c.norm(dim=-1) * t_c.norm(dim=-1)).clamp(min=1e-8))
            .mean()
            .item()
        )

        # Peak cross-correlation lag (in samples)
        xcorr = torch.fft.irfft(
            torch.fft.rfft(flat_true, n=2 * T) * torch.fft.rfft(flat_pred, n=2 * T).conj(),
            n=2 * T,
        )  # [B*L, 2T]
        lags = torch.fft.fftfreq(2 * T, d=1.0 / (2 * T)).long().to(xcorr.device)
        peak_lag = lags[xcorr.argmax(dim=-1)].float().mean().item()

        return {
            "val/neural/r2": r2,
            "val/neural/pearson": pearson,
            "val/neural/lag_samples": peak_lag,
        }

    def _plot_and_log_predictions(
        self,
        pred_bold,
        true_bold,
        pred_neural,
        true_neural,
        source_layer,
        source_pos,
        num_sources,
        voxel_pos=None,  # [B, 2] optional -- (h, w) actually plotted, for the suptitle below
        is_source_voxel=None,  # [B] bool optional -- whether voxel_pos is a true source
    ):
        """Per-sample BOLD+neural plot (`plot_neural_bold_layers`) for every
        sample in the batch, logged as `media/predictions` with `commit=False`
        -- relies on a later `commit=True` call (`_plot_and_log_latents`) in
        the same validation epoch to flush this into the same logged row; see
        `mich.MICH.on_validation_epoch_end`'s comment on why that matters for
        W&B history alignment.

        Args:
            pred_bold, true_bold, pred_neural, true_neural: [B, L, T].
            source_layer, source_pos, num_sources: Per-sample source metadata,
                forwarded to `plot_neural_bold_layers` (see its docstring).
            voxel_pos, is_source_voxel: If both given, each plot's suptitle
                names which (h, w) was actually plotted and whether it's a
                true source voxel; omitted (no suptitle) if either is None.
        """
        adapter = getattr(self, "_adapter", None)
        images = []
        for i in range(pred_bold.shape[0]):
            suptitle = None
            if voxel_pos is not None and is_source_voxel is not None:
                h, w = int(voxel_pos[i, 0]), int(voxel_pos[i, 1])
                kind = "Source" if bool(is_source_voxel[i]) else "Off-source"
                suptitle = f"{kind} voxel @ ({h}, {w})"
            image = plot_neural_bold_layers(
                pred_bold=pred_bold[i],
                true_bold=true_bold[i],
                pred_neural=pred_neural[i],
                true_neural=true_neural[i],
                source_layer=source_layer[i],
                source_pos=source_pos[i],
                num_sources=num_sources[i],
                suptitle=suptitle,
            )
            images.append(image)
        if adapter is not None and images:
            adapter.log(
                {"global_step": self.global_step, "media/predictions": images},
                commit=False,
            )
        for image in images:
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
        pred_v_star=None,
        true_v_star=None,
        pred_q_star=None,
        true_q_star=None,
        voxel_pos=None,  # [B, 2] optional -- (h, w) actually plotted, for the title below
        is_source_voxel=None,  # [B] bool optional -- whether voxel_pos is a true source
    ):
        """Per-sample latent-state plot (`plot_latent_layers`) for every sample,
        logged as `media/latents` with `commit=True` -- flushes this step's
        logged row (see `_plot_and_log_predictions`'s docstring on why the
        commit flag matters here).

        Args:
            pred_s, true_s, pred_f, true_f, pred_v, true_v, pred_q, true_q: [B, L, T].
            pred_v_star, true_v_star, pred_q_star, true_q_star: [B, L, T] each,
                or all four None (drain mode is all-or-nothing here, forwarded
                to `plot_latent_layers` accordingly).
            voxel_pos, is_source_voxel: As in `_plot_and_log_predictions`, but
                for each plot's title rather than a suptitle.
        """
        adapter = getattr(self, "_adapter", None)
        images = []
        for i in range(pred_s.shape[0]):
            title = "Latent States"
            if voxel_pos is not None and is_source_voxel is not None:
                h, w = int(voxel_pos[i, 0]), int(voxel_pos[i, 1])
                kind = "Source" if bool(is_source_voxel[i]) else "Off-source"
                title = f"Latent States -- {kind} voxel @ ({h}, {w})"
            if pred_v_star is not None:
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
                    title=title,
                )
            else:
                image = plot_latent_layers(
                    pred_f=pred_f[i],
                    true_f=true_f[i],
                    pred_s=pred_s[i],
                    true_s=true_s[i],
                    pred_v=pred_v[i],
                    true_v=true_v[i],
                    pred_q=pred_q[i],
                    true_q=true_q[i],
                    title=title,
                )
            images.append(image)
        if adapter is not None and images:
            adapter.log(
                {"global_step": self.global_step, "media/latents": images},
                commit=True,
            )
        for image in images:
            plt.close(image)

    def on_after_backward(self):
        """PL hook: flush `self._pending_train_log` (built by
        `mich.MICH._shared_step` this step) to `self._adapter`, appending FiLM
        and output-head gradient-norm metrics first -- run here rather than in
        `_shared_step` because gradients don't exist until after `backward()`.

        No-op if there's no pending log (e.g. this wasn't a training step) or
        no adapter (non-W&B/MLflow logger, or a rank `on_fit_start` didn't set
        one up for). Gradient norms are skipped at `global_step == 0` (no
        `.grad` yet on the very first step) and, per parameter group, only
        include parameters that currently have a non-None `.grad`.
        """
        pending = getattr(self, "_pending_train_log", None)
        self._pending_train_log = None
        if pending is None:
            return

        adapter = getattr(self, "_adapter", None)
        if adapter is None:
            return

        if self.global_step != 0:
            # FiLM linear vs output layer grad norms
            decoder = self.heinzle_net.spatial_decoder
            film = decoder.time_film
            linear_norms = [p.grad.norm() for p in film.linear.parameters() if p.grad is not None]
            out_norms = [p.grad.norm() for p in film.out.parameters() if p.grad is not None]
            if linear_norms:
                pending["gradients/film_linear_norm"] = torch.stack(linear_norms).norm().item()
            if out_norms:
                pending["gradients/film_out_norm"] = torch.stack(out_norms).norm().item()
            all_film_norms = linear_norms + out_norms
            if all_film_norms:
                pending["gradients/film_grad_norm"] = torch.stack(all_film_norms).norm().item()

            # Output head grad norms
            head_norms = [
                p.grad.norm()
                for head in decoder.out_heads
                for p in head.parameters()
                if p.grad is not None
            ]
            if head_norms:
                pending["gradients/out_heads_norm"] = torch.stack(head_norms).norm().item()

        adapter.log(pending)

    def on_fit_start(self) -> None:
        """PL hook: set `self._adapter` for this rank (see `make_run_adapter`),
        or leave it unset if the trainer's logger is neither W&B nor MLflow --
        every other method here treats a missing `self._adapter` as "logging
        disabled" via `getattr(self, "_adapter", None)`, not as an error."""
        if not isinstance(self.trainer.logger, (WandbLogger, MLFlowLogger)):
            return
        self._adapter = make_run_adapter(self.trainer, self.global_rank)
        self._adapter.configure_step_metric()

    def on_fit_end(self) -> None:
        """PL hook: finish and clear this rank's adapter, but only on non-zero
        ranks. Rank 0's adapter wraps the trainer's own primary logger
        connection (`trainer.logger`'s underlying W&B/MLflow run), which
        Lightning finishes itself; non-zero ranks' adapters wrap a separate,
        Lightning-invisible run (see `make_run_adapter`) that would otherwise
        never be closed.
        """
        adapter = getattr(self, "_adapter", None)
        if not self.trainer.is_global_zero and adapter is not None:
            adapter.finish()
            self._adapter = None
