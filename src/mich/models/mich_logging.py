"""Validation-time metrics, plotting, and rank-run/gradient-norm hooks for MICH."""

from __future__ import annotations

import matplotlib.pyplot as plt
import torch
from pytorch_lightning.loggers import MLFlowLogger, WandbLogger

from mich.utils.plotting import plot_latent_layers, plot_neural_bold_layers
from mich.utils.run_adapters import gpu_stats, make_run_adapter


class MICHLoggingMixin:
    """Neural-recovery metrics, plot assembly, and the PL hooks that manage per-rank
    runs and gradient-norm logging. Pure side effects -- no loss/physics math here."""

    @staticmethod
    def _neural_recovery_metrics(
        pred: torch.Tensor,  # [B, L, T]
        true: torch.Tensor,  # [B, L, T]
    ) -> dict[str, float]:
        """R2, Pearson r, and peak cross-correlation lag averaged over samples and layers."""
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

    def _plot_and_log_x_recon(
        self, pred_neural, pred_x_recon, true_x_recon, true_neural, source_layer, num_sources
    ):
        adapter = getattr(self, "_adapter", None)
        layer_names = ["Deep", "Middle", "Superficial"]
        images = []
        for i in range(pred_neural.shape[0]):
            n_layers = pred_neural.shape[1]
            valid_layers = source_layer[i, : int(num_sources[i])]
            fig, axes = plt.subplots(1, n_layers, figsize=(8 * n_layers, 8))
            if n_layers == 1:
                axes = [axes]
            for layer_index, ax in enumerate(axes):
                T = pred_neural.shape[2]
                t_full = torch.arange(T).float()
                t_short = torch.arange(T - 1).float()
                ax.plot(
                    t_full, true_neural[i, layer_index].cpu().float(), label="True x", color="green"
                )
                ax.plot(
                    t_full,
                    pred_neural[i, layer_index].cpu().float(),
                    label="Pred x (head)",
                    color="purple",
                    linestyle="--",
                )
                ax.plot(
                    t_short,
                    pred_x_recon[i, layer_index].cpu().float(),
                    label="Pred x (recon from s/f)",
                    color="orange",
                    linestyle=":",
                )
                ax.plot(
                    t_short,
                    true_x_recon[i, layer_index].cpu().float(),
                    label="True x (recon from s/f)",
                    color="blue",
                    linestyle=":",
                )
                n_src_here = int((valid_layers == layer_index).sum())
                ax.set_title(
                    f"{layer_names[layer_index]}"
                    + (f" [{n_src_here} src]" if n_src_here > 0 else "")
                )
                ax.legend(fontsize=6)
            fig.suptitle("x: head vs ODE reconstruction")
            fig.tight_layout()
            images.append(fig)
        if adapter is not None and images:
            adapter.log(
                {"global_step": self.global_step, "media/x_recon": images},
                commit=False,
            )
        for fig in images:
            plt.close(fig)

    def _plot_and_log_predictions(
        self,
        pred_bold,
        true_bold,
        pred_neural,
        true_neural,
        source_layer,
        source_pos,
        num_sources,
    ):
        adapter = getattr(self, "_adapter", None)
        images = []
        for i in range(pred_bold.shape[0]):
            image = plot_neural_bold_layers(
                pred_bold=pred_bold[i],
                true_bold=true_bold[i],
                pred_neural=pred_neural[i],
                true_neural=true_neural[i],
                source_layer=source_layer[i],
                source_pos=source_pos[i],
                num_sources=num_sources[i],
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
    ):
        adapter = getattr(self, "_adapter", None)
        images = []
        for i in range(pred_s.shape[0]):
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
                    title="Latent States",
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
                    title="Latent States",
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

        pending.update(gpu_stats())
        adapter.log(pending)

    def on_fit_start(self) -> None:
        if not isinstance(self.trainer.logger, (WandbLogger, MLFlowLogger)):
            return
        self._adapter = make_run_adapter(self.trainer, self.global_rank)
        self._adapter.configure_step_metric()

    def on_fit_end(self) -> None:
        adapter = getattr(self, "_adapter", None)
        if not self.trainer.is_global_zero and adapter is not None:
            adapter.finish()
            self._adapter = None
