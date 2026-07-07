"""Data/physics/supervision loss computation for MICH, plus the Gaussian-PSF blur."""

from __future__ import annotations

from functools import partial
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from mich.data.balloon import AcquisitionConstants, PointSpreadFunction
from mich.models.collocation import CollocationMixin


class MICHLossMixin(CollocationMixin):
    """Data loss, physics (ODE-residual) loss, and supervision loss.

    Inherits CollocationMixin (index sampling/gathering) for static analysis of the
    `self._gather_*`/`self._signal_index` calls below; also depends on
    LearnablePhysioMixin (self._physio / self._current_acquisition) being mixed in
    alongside this one at the concrete-model level.
    """

    # Mapping from z_hat signal name -> batch key
    _SUPERVISION_KEYS_FULL = (
        ("s", "s"),
        ("f", "f"),
        ("v", "v"),
        ("q", "q"),
        ("vstar", "v_star"),
        ("qstar", "q_star"),
    )
    _SUPERVISION_KEYS_SINGLE = (
        ("s", "s"),
        ("f", "f"),
        ("v", "v"),
        ("q", "q"),
    )

    def _setup_psf(self) -> None:
        """Build PSF objects and register 2D kernels as buffers so they move with the device."""
        psf_fwhm = getattr(self.hparams, "psf_fwhm", None)
        if psf_fwhm is None:
            self._psf = None
            return
        self._psf = [PointSpreadFunction(fwhm=f) for f in psf_fwhm]
        for i, (fwhm, psf) in enumerate(zip(psf_fwhm, self._psf, strict=True)):
            if fwhm is not None and fwhm > 0:
                kernel = torch.as_tensor(psf.kernel_2d(), dtype=torch.float32)
            else:
                kernel = torch.tensor([[1.0]], dtype=torch.float32)
            self.register_buffer(f"_psf_kernel_{i}", kernel)

    def _apply_psf_blur(self, bold: torch.Tensor) -> torch.Tensor:
        """Blur each layer of `bold` [B, L, T, H, W] with its per-layer Gaussian PSF kernel."""
        if self._psf is None:
            return bold
        B_size, L_size, T_size, H_size, W_size = bold.shape
        layers_blurred = []
        for i in range(L_size):
            kernel = getattr(self, f"_psf_kernel_{i}").to(bold.device)
            pad = kernel.shape[-1] // 2
            x = bold[:, i].reshape(B_size * T_size, 1, H_size, W_size)
            layers_blurred.append(
                F.conv2d(x, kernel, padding=pad)
                .reshape(B_size, T_size, H_size, W_size)
                .to(bold.dtype)
            )
        return torch.stack(layers_blurred, dim=1)  # [B, L, T, H, W]

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
    def _compute_bold(
        v: torch.Tensor, q: torch.Tensor, acquisition: AcquisitionConstants, V0: float
    ) -> torch.Tensor:
        k1, k2, k3 = acquisition.k1, acquisition.k2, acquisition.k3
        return V0 * (k1 * (1 - q) + k2 * (1 - q / v) + k3 * (1 - v))

    @staticmethod
    def _compute_bold_at(
        z_hat: torch.Tensor, idx, acquisition: AcquisitionConstants, V0: float
    ) -> torch.Tensor:
        v = CollocationMixin._gather_z_hat_at(z_hat, idx, signal="v")
        q = CollocationMixin._gather_z_hat_at(z_hat, idx, signal="q")
        return MICHLossMixin._compute_bold(v, q, acquisition, V0)

    def _antisteady_loss(
        self,
        z_hat: torch.Tensor,  # [B, 7, L, T, H, W]
        source_position: torch.Tensor,  # [B, S, 2]
        source_layer: torch.Tensor,  # [B, S]
        num_sources: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        """Two-sided variance penalty at each source voxel.

        Positive term: at a source's own (layer, h, w), x should show real
        dynamics over time (not settle to steady state).
        Negative term: at that SAME (h, w) in every OTHER layer, there is no
        neural source -- so despite possible BOLD drainage from the source
        layer, x should stay flat. This teaches the model that BOLD can
        exist without co-located neural activity (venous drainage) rather
        than hallucinating neural signal wherever BOLD moves.
        """
        B, _, L = z_hat.shape[:3]
        S = source_position.shape[1]
        device = z_hat.device

        mask = torch.arange(S, device=device)[None, :] < num_sources[:, None]  # [B, S]
        b_idx = torch.arange(B, device=device)[:, None].expand(B, S)  # [B, S]
        src_h = source_position[..., 0].long()  # [B, S]
        src_w = source_position[..., 1].long()  # [B, S]
        src_l = source_layer.clamp(min=0, max=L - 1).long()  # [B, S], padding clamped, masked below

        x_idx = self._signal_index("x")
        eps = getattr(self.hparams.loss_config, "antisteady_epsilon", 0.01)

        # x at each source's own voxel/layer: [B, S, T]
        x_src = z_hat[b_idx, x_idx, src_l, :, src_h, src_w]
        x_var = x_src.var(dim=-1)  # [B, S]
        pos_term = F.relu(eps - x_var)
        pos_loss = (pos_term * mask).sum() / mask.sum().clamp(min=1)

        # x at the same (h, w) across every OTHER layer: [B, S, L, T]
        x_all_layers = z_hat[b_idx, x_idx, :, :, src_h, src_w]
        x_var_other = x_all_layers.var(dim=-1)  # [B, S, L]
        other_layer_mask = torch.arange(L, device=device)[None, None, :] != src_l[:, :, None]
        other_mask = mask[:, :, None] & other_layer_mask  # [B, S, L]
        eps_neg = getattr(self.hparams.loss_config, "antisteady_neg_epsilon", eps)
        neg_term = F.relu(x_var_other - eps_neg)
        neg_loss = (neg_term * other_mask).sum() / other_mask.sum().clamp(min=1)

        lambda_neg = getattr(self.hparams.loss_config, "lambda_antisteady_neg", 1.0)
        return pos_loss + lambda_neg * neg_loss

    def _data_loss(
        self,
        z_hat: torch.Tensor,
        bold_norm: torch.Tensor,
        source_position: torch.Tensor | None = None,
        num_sources: torch.Tensor | None = None,
    ) -> torch.Tensor:
        collocation = self._sample_collocation_indices(
            T=bold_norm.shape[2],
            H=bold_norm.shape[3],
            W=bold_norm.shape[4],
            n_times=self.hparams.loss_config.n_time,
            n_space=self.hparams.loss_config.n_space,
            device=z_hat.device,
            source_position=source_position,
            num_sources=num_sources,
            dense_spatial_frac=self.hparams.loss_config.dense_spatial_frac,
            dense_spatial_radius=self.hparams.loss_config.dense_spatial_radius,
            dense_time_frac=self.hparams.loss_config.dense_time_frac,
            dense_time_lo=self.hparams.loss_config.dense_time_lo,
            dense_time_hi=self.hparams.loss_config.dense_time_hi,
            uniform_time_lo=self.hparams.loss_config.uniform_time_lo,
        )

        v_idx, q_idx = self._signal_index("v"), self._signal_index("q")
        pred_v = z_hat[:, v_idx]
        pred_q = z_hat[:, q_idx]
        pred_bold = self._compute_bold(
            pred_v, pred_q, acquisition=self._current_acquisition(), V0=self._physio("V0")
        )
        pred_bold = self._apply_psf_blur(pred_bold)  # [B, L, T, H, W]

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

        # Source voxel loss -- full T, all layers, per valid source; shape per layer:
        # [M, T] where M = total valid sources across the batch; Pearson over T (dim=1)
        B = pred_bold.shape[0]
        S = source_position.shape[1]
        b_idx = torch.arange(B, device=pred_bold.device)[:, None].expand(B, S)
        src_h = source_position[..., 0].long()  # [B, S]
        src_w = source_position[..., 1].long()
        pred_bold_src = pred_bold[b_idx, :, :, src_h, src_w]  # [B, S, L, T]
        true_bold_src = true_bold[b_idx, :, :, src_h, src_w]  # [B, S, L, T]
        T_src = pred_bold_src.shape[-1]

        mask = torch.arange(S, device=pred_bold.device)[None, :] < num_sources[:, None]  # [B, S]
        pred_bold_src = pred_bold_src.reshape(B * S, L, T_src)[mask.reshape(-1)]  # [M, L, T]
        true_bold_src = true_bold_src.reshape(B * S, L, T_src)[mask.reshape(-1)]
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
        idx,
        layer: int,
        burn_in: int,
        order: str,
    ) -> Mapping[str, torch.Tensor]:
        has_drain = z_hat.shape[1] > 5  # vstar/qstar only present in multi-layer mode

        x = self._gather_z_hat_at(z_hat, idx, signal="x")[:, layer]
        s = self._gather_z_hat_at(z_hat, idx, signal="s")[:, layer]
        f = self._gather_z_hat_at(z_hat, idx, signal="f")[:, layer]
        v = self._gather_z_hat_at(z_hat, idx, signal="v")[:, layer]
        q = self._gather_z_hat_at(z_hat, idx, signal="q")[:, layer]

        state_dict = {"x": x, "s": s, "f": f, "v": v, "q": q}
        if has_drain:
            v_star = self._gather_z_hat_at(z_hat, idx, signal="vstar")[:, layer]
            q_star = self._gather_z_hat_at(z_hat, idx, signal="qstar")[:, layer]
            state_dict.update({"vstar": v_star, "qstar": q_star})

        states = self._sanitise_states(state_dict)
        x, s, f, v, q = states["x"], states["s"], states["f"], states["v"], states["q"]
        if has_drain:
            v_star, q_star = states["vstar"], states["qstar"]

        s_scale = 1.0
        f_scale = 1.0
        v_scale = 1.0
        q_scale = 1.0

        alpha = self._physio("alpha")
        gamma = self._physio("gamma")
        kappa = self._physio("kappa")
        lambda_d = self.hparams.haemo.lambda_d  # not learnable (currently out of scope)
        tau = self._physio("tau")
        tau_d = self.hparams.haemo.tau_d  # not learnable (currently out of scope)
        E0 = self._physio("E0")

        ds_dt = self._gather_grad_at(dz_hat_dt, layer, idx, signal="s")
        s_target = x - kappa * s - gamma * (f - 1)
        s_loss = self._ode_loss_fn(ds_dt[:, burn_in:] / s_scale, s_target[:, burn_in:] / s_scale)

        df_dt = self._gather_grad_at(dz_hat_dt, layer, idx, signal="f")
        f_loss = self._ode_loss_fn(df_dt[:, burn_in:] / f_scale, s[:, burn_in:] / f_scale)

        dv_dt = self._gather_grad_at(dz_hat_dt, layer, idx, signal="v")
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
        else:
            raise ValueError(
                f"Expected order to be one of `linear`, `quadractic` or `exact`. But recieved {order}"
            )
        if has_drain and layer > 0:
            vstar_deeper = self._gather_z_hat_at(z_hat, idx, signal="vstar")[:, layer - 1]
            target_vdot += lambda_d * vstar_deeper
        v_loss = self._ode_loss_fn(
            dv_dt[:, burn_in:] / v_scale,
            (target_vdot[:, burn_in:] / tau) / v_scale,
        )

        dq_dt = self._gather_grad_at(dz_hat_dt, layer, idx, signal="q")
        if order == "exact":
            target_qdot = f * (1 - (1 - E0) ** (1 / f)) / E0 - q * v ** (1 / alpha - 1)
        elif order == "linear":
            f, v, q = f - 1, v - 1, q - 1
            log_1mE0 = torch.log(torch.as_tensor(1 - E0, dtype=v.dtype, device=v.device))
            beta_1 = (1 - E0) * log_1mE0 / E0
            target_qdot = (1 + beta_1) * f - q - (1 / alpha - 1) * v
            f, v, q = f + 1, v + 1, q + 1
        elif order == "quadratic":
            f, v, q = f - 1, v - 1, q - 1
            log_1mE0 = torch.log(torch.as_tensor(1 - E0, dtype=v.dtype, device=v.device))
            beta_1 = (1 - E0) * log_1mE0 / E0
            beta_2 = beta_1 * log_1mE0 / 2
            target_qdot = (
                (1 + beta_1) * f
                - q
                - (1 / alpha - 1) * v
                - beta_2 * f**2
                - (1 / alpha - 1) * v * q
                - (1 / 2) * (1 / alpha - 1) * (1 / alpha - 2) * v**2
            )
            f, v, q = f + 1, v + 1, q + 1
        if has_drain and layer > 0:
            qstar_deeper = self._gather_z_hat_at(z_hat, idx, signal="qstar")[:, layer - 1]
            target_qdot += lambda_d * qstar_deeper
        q_loss = self._ode_loss_fn(
            dq_dt[:, burn_in:] / q_scale,
            (target_qdot[:, burn_in:] / tau) / q_scale,
        )

        losses = {"s": s_loss, "f": f_loss, "v": v_loss, "q": q_loss}

        if has_drain:
            dv_star_dt = self._gather_grad_at(dz_hat_dt, layer, idx, signal="vstar")
            v_star_target = (-v_star + v - 1) / tau_d
            losses["vstar"] = self._ode_loss_fn(dv_star_dt[:, burn_in:], v_star_target[:, burn_in:])
            dq_star_dt = self._gather_grad_at(dz_hat_dt, layer, idx, signal="qstar")
            q_star_target = (-q_star + q - 1) / tau_d
            losses["qstar"] = self._ode_loss_fn(dq_star_dt[:, burn_in:], q_star_target[:, burn_in:])

        return losses

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

    def _physics_loss(  # add a parameter that says whether the loss is linear or nonlinear
        self,
        z_hat: torch.Tensor,
        dz_hat_dt: torch.Tensor,
        order: str,
        lambda_smooth: float = 0.0,
        source_position: torch.Tensor | None = None,
        num_sources: torch.Tensor | None = None,
    ) -> torch.Tensor:
        idx = self._sample_collocation_indices(
            T=z_hat.shape[3],
            H=z_hat.shape[4],
            W=z_hat.shape[5],
            n_times=self.hparams.loss_config.n_time,
            n_space=self.hparams.loss_config.n_space,
            device=z_hat.device,
            source_position=source_position,
            num_sources=num_sources,
            dense_spatial_frac=self.hparams.loss_config.dense_spatial_frac,
            dense_spatial_radius=self.hparams.loss_config.dense_spatial_radius,
            dense_time_frac=self.hparams.loss_config.dense_time_frac,
            dense_time_lo=self.hparams.loss_config.dense_time_lo,
            dense_time_hi=self.hparams.loss_config.dense_time_hi,
            uniform_time_lo=self.hparams.loss_config.uniform_time_lo,
        )
        has_drain = z_hat.shape[1] > 5
        _eq_keys = ("s", "f", "v", "q", "vstar", "qstar") if has_drain else ("s", "f", "v", "q")
        n_eq = len(_eq_keys)
        tot_physics_loss = torch.tensor(0.0, device=z_hat.device, dtype=torch.float32)
        per_eq = {k: torch.tensor(0.0, device=z_hat.device, dtype=torch.float32) for k in _eq_keys}
        n_layers = z_hat.shape[2]
        for layer in range(n_layers):
            layer_losses = self._compute_physics_layer_loss(
                z_hat,
                dz_hat_dt,
                idx,
                layer=layer,
                burn_in=self.hparams.loss_config.burn_in,
                order=order,
            )
            layer_total = sum(layer_losses.values()).float() / n_eq
            tot_physics_loss = tot_physics_loss + layer_total / n_layers
            for k in _eq_keys:
                per_eq[k] = per_eq[k] + layer_losses[k].float() / n_layers

        # Smoothness of gradients
        if lambda_smooth <= 0:
            return tot_physics_loss, per_eq
        else:
            dz_dt_fd = z_hat[:, :, :, 1:] - z_hat[:, :, :, :-1]  # [B, S, L, T-1, H, W]
            smoothness_loss = dz_dt_fd.pow(2).mean()
            return tot_physics_loss + lambda_smooth * smoothness_loss, per_eq

    def _supervision_keys(self, z_hat: torch.Tensor):
        keys = self._SUPERVISION_KEYS_SINGLE if z_hat.shape[1] <= 5 else self._SUPERVISION_KEYS_FULL
        # TEMPORARY ablation switch: direct supervision on x (batch["neural"]) is not
        # part of the normal MICH objective (x is meant to be recovered purely via the
        # physics residual against the well-supervised s-trajectory, since real fMRI has
        # no ground-truth neural signal to supervise against). Only enable to diagnose
        # whether varying block-amplitude neural signals are an identifiability/training
        # issue vs. an architecture-capacity issue.
        if getattr(self.hparams.loss_config, "supervise_x", False):
            keys = (("x", "neural"), *keys)
        return keys

    def _supervision_loss(
        self,
        z_hat: torch.Tensor,  # [B, 7, L, T, H, W]
        batch: dict,
        source_position: torch.Tensor | None = None,
        num_sources: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """MSE between predicted and ground-truth latent states at collocation points."""
        # Use T_min so collocation indices are valid for both z_hat and true latents.
        T_latent = batch["s"].shape[2]
        T_min = min(z_hat.shape[3], T_latent)
        lc = self.hparams.loss_config
        idx = self._sample_collocation_indices(
            T=T_min,
            H=z_hat.shape[4],
            W=z_hat.shape[5],
            n_times=lc.n_time,
            n_space=lc.n_space,
            device=z_hat.device,
            source_position=source_position,
            num_sources=num_sources,
            dense_spatial_frac=lc.dense_spatial_frac,
            dense_spatial_radius=lc.dense_spatial_radius,
            dense_time_frac=lc.dense_time_frac,
            dense_time_lo=lc.dense_time_lo,
            dense_time_hi=lc.dense_time_hi,
            uniform_time_lo=lc.uniform_time_lo,
        )
        per_sig: dict[str, torch.Tensor] = {}
        for sig, bk in self._supervision_keys(z_hat):
            true = batch[bk].float()  # [B, L, T_latent, H, W]
            pred_at = self._gather_z_hat_at(z_hat, idx, signal=sig).float()  # [B, L, n_t, n_s]
            true_at = self._gather_bold_at(true, idx).float()  # [B, L, n_t, n_s]
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
        source_position: torch.Tensor,  # [B, S, 2]
        num_sources: torch.Tensor,  # [B]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """MSE between predicted and ground-truth latent states at each valid source voxel, across all T."""
        B = z_hat.shape[0]
        S = source_position.shape[1]
        device = z_hat.device
        b_idx = torch.arange(B, device=device)[:, None].expand(B, S)
        src_h = source_position[..., 0].long()  # [B, S]
        src_w = source_position[..., 1].long()
        mask = torch.arange(S, device=device)[None, :] < num_sources[:, None]  # [B, S]
        T_latent = batch["s"].shape[2]
        T_pred = z_hat.shape[3]
        T_min = min(T_pred, T_latent)

        per_sig: dict[str, torch.Tensor] = {}
        for sig, bk in self._supervision_keys(z_hat):
            true = batch[bk].float()  # [B, L, T_latent, H, W]
            pred = z_hat[:, self._signal_index(sig)].float()  # [B, L, T, H, W]
            pred_src = pred[b_idx, :, :T_min, src_h, src_w]  # [B, S, L, T_min]
            true_src = true[b_idx, :, :T_min, src_h, src_w]  # [B, S, L, T_min]
            L = pred_src.shape[2]
            pred_src = pred_src.reshape(B * S, L, T_min)[mask.reshape(-1)]  # [M, L, T_min]
            true_src = true_src.reshape(B * S, L, T_min)[mask.reshape(-1)]
            per_sig[sig] = torch.stack(
                [
                    self._supervision_loss_fn(pred_src[:, layer_idx], true_src[:, layer_idx])
                    for layer_idx in range(L)
                ]
            ).mean()

        total = sum(per_sig.values()) / len(per_sig)
        return total, per_sig

    _DSDT_BATCH_KEY = {"s": "s", "f": "f", "v": "v", "q": "q", "vstar": "v_star", "qstar": "q_star"}

    def _derivative_supervision_loss(
        self,
        dz_hat_dt: torch.Tensor,  # [B, 7, L, T, H, W]
        batch: dict,
        source_position: torch.Tensor,  # [B, S, 2]
        num_sources: torch.Tensor,  # [B]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """MSE between the network's analytic dz_hat/dt and a finite-difference estimate
        of the same derivative from ground-truth latents, at each valid source voxel
        across all T.

        Motivation: value-only supervision (_source_supervision_loss) barely penalises a
        small timing/phase error -- a slightly-lagged copy of a smooth curve has near
        identical MSE/Pearson. x is only ever recovered via ds/dt through the physics
        residual (there's no ground truth to check it against directly), so any phase lag
        in the learned s(t) propagates straight into x with nothing to correct it.
        Matching ds/dt directly is sharply sensitive to timing, unlike value-matching.

        Compared in per-index units (i.e. per stored sample, not per unit of the model's
        normalised [0, 1] time grid): dz_hat_dt is d(z_hat)/d(t_norm), so it's divided by
        (T_min - 1) to undo that normalisation rather than scaling the (small, O(1))
        ground-truth finite difference up to match it -- avoids inflating the loss target
        to ~(T_min - 1) times every other signal's scale at sharp transitions, which
        previously produced squared-error gradients large enough to destabilise training.
        """
        lc = self.hparams.loss_config
        signals = tuple(getattr(lc, "dzdt_supervision_signals", ("s",)))
        B = dz_hat_dt.shape[0]
        S = source_position.shape[1]
        device = dz_hat_dt.device
        b_idx = torch.arange(B, device=device)[:, None].expand(B, S)
        src_h = source_position[..., 0].long()  # [B, S]
        src_w = source_position[..., 1].long()
        mask = torch.arange(S, device=device)[None, :] < num_sources[:, None]  # [B, S]

        per_sig: dict[str, torch.Tensor] = {}
        for sig in signals:
            bk = self._DSDT_BATCH_KEY[sig]
            true = batch[bk].float()  # [B, L, T_latent, H, W]
            pred = dz_hat_dt[:, self._signal_index(sig)].float()  # [B, L, T, H, W]
            T_min = min(pred.shape[2], true.shape[2])
            true = true[:, :, :T_min]
            # Per-index finite difference (NOT rescaled to the model's per-normalised-time
            # convention) -- rescaling the *target* up by (T_min - 1) (~= 100 for this
            # simulation's dt=1.0/time_duration=100) inflates it far past every other
            # signal's O(1) scale, so squared error against it dwarfs the rest of the loss
            # and blows up training. Instead bring the *prediction* down to the same
            # per-index scale below; identical constraint, ~(T_min-1)^2 smaller gradient.
            true_dt = torch.gradient(true, dim=2)[0]

            pred_src = pred[b_idx, :, :T_min, src_h, src_w] / (T_min - 1)  # [B, S, L, T_min]
            true_src = true_dt[b_idx, :, :T_min, src_h, src_w]  # [B, S, L, T_min]
            L = pred_src.shape[2]
            pred_src = pred_src.reshape(B * S, L, T_min)[mask.reshape(-1)]  # [M, L, T_min]
            true_src = true_src.reshape(B * S, L, T_min)[mask.reshape(-1)]
            per_sig[sig] = torch.stack(
                [
                    self._dzdt_loss_fn(pred_src[:, layer_idx], true_src[:, layer_idx])
                    for layer_idx in range(L)
                ]
            ).mean()

        total = sum(per_sig.values()) / len(per_sig)
        return total, per_sig
