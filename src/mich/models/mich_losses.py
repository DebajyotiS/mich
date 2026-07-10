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

    def _balloon_v_q_dot_targets(
        self, f: torch.Tensor, v: torch.Tensor, q: torch.Tensor, order: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Un-normalised (not yet divided by tau) dv/dt, dq/dt targets from the balloon
        ODE's v/q equations (order in {exact, linear, quadratic}). Shared by
        _compute_physics_layer_loss (evaluated on model states) and
        _derivative_supervision_loss (evaluated on ground-truth states) so the two
        formulas can't drift out of sync with each other.
        """
        alpha = self._physio("alpha")
        E0 = self._physio("E0")

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

        return target_vdot, target_qdot

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

        # dz_hat_dt is d(z_hat)/d(t_norm) (the decoder's analytic derivative w.r.t. its
        # [0,1]-normalised time grid), but every RHS below (x - kappa*s - gamma*(f-1), s,
        # target_vdot/tau, target_qdot/tau) is a plain physical-value combination with no
        # t_norm involved -- i.e. already in per-sample-index ("physical") units. Dividing
        # the *derivative* term by (T-1) converts it to that same per-index convention;
        # matches _derivative_supervision_loss's identical `pred_src / (T_min - 1)`. Only
        # the derivative side gets divided
        total_time_samples = z_hat.shape[3]
        t_norm_to_physical = total_time_samples - 1

        alpha = self._physio("alpha")
        gamma = self._physio("gamma")
        kappa = self._physio("kappa")
        lambda_d = self.hparams.haemo.lambda_d  # not learnable (currently out of scope)
        tau = self._physio("tau")
        tau_d = self.hparams.haemo.tau_d  # not learnable (currently out of scope)
        E0 = self._physio("E0")

        ds_dt = self._gather_grad_at(dz_hat_dt, layer, idx, signal="s") / t_norm_to_physical
        s_target = x - kappa * s - gamma * (f - 1)
        s_loss = self._ode_loss_fn(ds_dt[:, burn_in:], s_target[:, burn_in:])

        df_dt = self._gather_grad_at(dz_hat_dt, layer, idx, signal="f") / t_norm_to_physical
        f_loss = self._ode_loss_fn(df_dt[:, burn_in:], s[:, burn_in:])

        dv_dt = self._gather_grad_at(dz_hat_dt, layer, idx, signal="v") / t_norm_to_physical
        target_vdot, target_qdot = self._balloon_v_q_dot_targets(f, v, q, order)
        if has_drain and layer > 0:
            vstar_deeper = self._gather_z_hat_at(z_hat, idx, signal="vstar")[:, layer - 1]
            target_vdot = target_vdot + lambda_d * vstar_deeper
        v_loss = self._ode_loss_fn(
            dv_dt[:, burn_in:],
            target_vdot[:, burn_in:] / tau,
        )

        dq_dt = self._gather_grad_at(dz_hat_dt, layer, idx, signal="q") / t_norm_to_physical
        if has_drain and layer > 0:
            qstar_deeper = self._gather_z_hat_at(z_hat, idx, signal="qstar")[:, layer - 1]
            target_qdot = target_qdot + lambda_d * qstar_deeper
        q_loss = self._ode_loss_fn(
            dq_dt[:, burn_in:],
            target_qdot[:, burn_in:] / tau,
        )

        losses = {"s": s_loss, "f": f_loss, "v": v_loss, "q": q_loss}

        if has_drain:
            dv_star_dt = (
                self._gather_grad_at(dz_hat_dt, layer, idx, signal="vstar") / t_norm_to_physical
            )
            v_star_target = (-v_star + v - 1) / tau_d
            losses["vstar"] = self._ode_loss_fn(dv_star_dt[:, burn_in:], v_star_target[:, burn_in:])
            dq_star_dt = (
                self._gather_grad_at(dz_hat_dt, layer, idx, signal="qstar") / t_norm_to_physical
            )
            q_star_target = (-q_star + q - 1) / tau_d
            losses["qstar"] = self._ode_loss_fn(dq_star_dt[:, burn_in:], q_star_target[:, burn_in:])

        return losses

    def _anneal_between(
        self, start_val: float, end_val: float, anneal_start_step: int, anneal_end_step: int
    ) -> float:
        """Linearly interpolate start_val -> end_val over [anneal_start_step,
        anneal_end_step], clamped to start_val before and end_val after."""
        if anneal_end_step <= anneal_start_step:
            return end_val
        frac = (self.global_step - anneal_start_step) / (anneal_end_step - anneal_start_step)
        frac = min(1.0, max(0.0, frac))
        return start_val + frac * (end_val - start_val)

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

    def _derivative_supervision_loss(
        self,
        dz_hat_dt: torch.Tensor,  # [B, 7, L, T, H, W]
        batch: dict,
        source_position: torch.Tensor,  # [B, S, 2]
        num_sources: torch.Tensor,  # [B]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """MSE between the network's analytic dz_hat/dt and the exact Balloon-Windkessel
        ODE derivative evaluated on ground-truth latents, at each valid source voxel
        across all T.

        Compared in per-index units (i.e. per stored sample, not per unit of the model's
        normalised [0, 1] time grid): dz_hat_dt is d(z_hat)/d(t_norm), so it's divided by
        (T_min - 1) to undo that normalisation, matching _compute_physics_layer_loss's
        identical convention.
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

        x_true = batch["neural"].float()  # [B, L, T_latent, H, W]
        s_true = batch["s"].float()
        f_true = batch["f"].float()
        v_true = batch["v"].float()
        q_true = batch["q"].float()
        T_min = min(dz_hat_dt.shape[3], s_true.shape[2])
        x_true, s_true, f_true, v_true, q_true = (
            t[:, :, :T_min] for t in (x_true, s_true, f_true, v_true, q_true)
        )

        kappa = self._physio("kappa")
        gamma = self._physio("gamma")
        tau = self._physio("tau")
        target_vdot, target_qdot = self._balloon_v_q_dot_targets(f_true, v_true, q_true, lc.order)
        # NOTE: no cross-layer drain coupling here (unlike _compute_physics_layer_loss's
        # vstar/qstar-deeper term) -- not needed for the currently-configured signals
        # (s, f, v, q); would need extending before enabling "vstar"/"qstar" here for a
        # multi-layer, lambda_d != 0 config.
        analytic_target = {
            "s": x_true - kappa * s_true - gamma * (f_true - 1.0),
            "f": s_true,
            "v": target_vdot / tau,
            "q": target_qdot / tau,
        }

        per_sig: dict[str, torch.Tensor] = {}
        for sig in signals:
            true_dt = analytic_target[sig]  # [B, L, T_min, H, W]
            pred = dz_hat_dt[:, self._signal_index(sig)].float()  # [B, L, T, H, W]

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

    def _x_phase_loss(
        self,
        z_hat: torch.Tensor,  # [B, 7, L, T, H, W]
        dz_hat_dt: torch.Tensor,  # [B, 7, L, T, H, W]
        source_position: torch.Tensor,  # [B, S, 2]
        num_sources: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        """Coherent, full-T, same-voxel phase-sensitive loss pulling x_hat toward its own
        physics-residual reconstruction x_rhs = Dp_s + kappa*s_hat + gamma*(f_hat-1).

        The physics residual (_compute_physics_layer_loss) is the only training signal
        touching x_hat's value at all, but it's evaluated at scattered collocation points
        -- a fresh random draw of times (and, separately, spatial locations) every step
        (see _sample_collocation_indices), never a coherent, ordered whole trajectory.
        s/f/v/q, by contrast, are judged by _source_supervision_loss on their full,
        ordered T-length trajectory at the fixed source voxel every step -- a whole-shape
        comparison, not a cloud of independent point constraints. This loss gives x_hat
        that same kind of coherent, per-step, whole-trajectory signal, against a
        reconstruction built entirely from data-derived quantities (Dp_s, s_hat, f_hat)
        rather than ground-truth x, so it stays usable with real fMRI (no x label needed).
        Uses hparams.loss_config.x_phase_loss (mse+pearson by default) so shape/phase
        misalignment is penalised directly, not just implicitly through pointwise MSE.

        Not detached: x_rhs is a function of s_hat/f_hat/Dp_s, so gradients flow both
        ways, same bidirectional-consistency philosophy as the physics residual itself.
        """
        B = z_hat.shape[0]
        S = source_position.shape[1]
        T = z_hat.shape[3]
        device = z_hat.device
        b_idx = torch.arange(B, device=device)[:, None].expand(B, S)
        src_h = source_position[..., 0].long()  # [B, S]
        src_w = source_position[..., 1].long()
        mask = torch.arange(S, device=device)[None, :] < num_sources[:, None]  # [B, S]

        kappa = self._physio("kappa")
        gamma = self._physio("gamma")
        burn_in = self.hparams.loss_config.burn_in
        t_norm_to_physical = T - 1

        x_hat = z_hat[:, self._signal_index("x")].float()  # [B, L, T, H, W]
        s_hat = z_hat[:, self._signal_index("s")].float()
        f_hat = z_hat[:, self._signal_index("f")].float()
        Dp_s = dz_hat_dt[:, self._signal_index("s")].float() / t_norm_to_physical

        x_src = x_hat[b_idx, :, :, src_h, src_w]  # [B, S, L, T]
        s_src = s_hat[b_idx, :, :, src_h, src_w]
        f_src = f_hat[b_idx, :, :, src_h, src_w]
        Dp_s_src = Dp_s[b_idx, :, :, src_h, src_w]

        x_rhs_src = Dp_s_src + kappa * s_src + gamma * (f_src - 1.0)

        L = x_src.shape[2]
        x_src = x_src.reshape(B * S, L, T)[mask.reshape(-1)]  # [M, L, T]
        x_rhs_src = x_rhs_src.reshape(B * S, L, T)[mask.reshape(-1)]

        return torch.stack(
            [
                self._x_phase_loss_fn(
                    x_src[:, layer_idx, burn_in:], x_rhs_src[:, layer_idx, burn_in:]
                )
                for layer_idx in range(L)
            ]
        ).mean()
