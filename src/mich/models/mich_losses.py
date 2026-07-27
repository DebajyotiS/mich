"""Data/physics/supervision loss computation for MICH, plus the Gaussian-PSF blur.

Governing equations (Heinzle/Balloon-Windkessel, `order="exact"`; see
`_balloon_v_q_dot_targets` for the `order="linear"`/`"quadratic"` Taylor expansions
about the resting state s=0, f=v=q=1):

    ds/dt = x - kappa*s - gamma*(f - 1)
    df/dt = s
    dv/dt = (f - v**(1/alpha)) / tau
    dq/dt = (f*(1 - (1-E0)**(1/f))/E0 - q*v**(1/alpha - 1)) / tau

with the drain-mode addition for layer > 0: `dv/dt += lambda_d * v_star_below /
tau`, `dq/dt += lambda_d * q_star_below / tau`, and the delay filter
`dv_star/dt = (-v_star + (v-1)) / tau_d` (same for q_star). This is the same math
`mich.data.balloon.balloon_derivatives`/`delay_filter_derivatives` implement for
generating simulated ground truth -- the two are independent implementations (one
numpy/scipy, one torch/autograd) and must be kept in sync by hand; there is no
shared source of truth between them.

The s-equation above (`ds/dt = x - kappa*s - gamma*(f-1)`) recurs across this
module -- as the physics-loss residual target in `_compute_physics_layer_loss`, as
the analytic target in `_derivative_supervision_loss`, as the reconstruction
`_x_phase_loss` pulls x_hat toward, and as the rationale for
`_quiescence_consistency_loss`'s gate. All four must agree with the equation above.

Time-derivative units convention ("t_norm" vs "physical"): `dz_hat_dt` (produced by
`blocks.SpatioTemporalDecoder`) is d(z_hat)/d(t_norm), the derivative with respect
to the model's [0, 1]-normalised time grid -- not with respect to the stored
per-sample index. Every ODE target above is a plain physical-value combination
with no t_norm involved, so any `dz_hat_dt` slice compared against one of those
targets must first be divided by `T - 1` (the grid's index-to-t_norm scale) to
convert it to the same per-index convention. `_compute_physics_layer_loss`,
`_derivative_supervision_loss`, and `_x_phase_loss` each do this via a local
`t_norm_to_physical = T - 1` divisor.

Source metadata convention, shared by every loss below that needs to locate the
known source(s) within a sample: `source_position` is `[B, S, 2]` (h, w) per
source slot, `source_layer` is `[B, S]` (that source's layer index), and
`num_sources` is `[B]` (how many of the S slots are real). S is padded to the
batch's max source count, so slots at index >= `num_sources[b]` hold undefined
values for sample b and must be excluded via a `torch.arange(S) < num_sources[:,
None]` mask before use, not read directly.
"""

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
                kernel = torch.tensor([[[[1.0]]]], dtype=torch.float32)
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

        Args:
            loss_cfg: A mapping-like config object (attribute access via
                `getattr`) with `type` ("mse"|"huber"|"pearson"|"mse+pearson"|
                "huber+pearson", default "mse"), `huber_delta` (default 1.0),
                `lambda_pearson` (default 1.0), and -- for the two combined
                types -- `lambda_npearson` (default 1.0). None is treated as
                `type="mse"` with all other defaults. Pearson correlation is
                computed over dim=1 (the time/sequence dimension).

        Warning:
            For the "mse+pearson"/"huber+pearson" types, `lambda_pearson` is
            read from `loss_cfg` once here, at build time, and closed over as a
            plain float; `lambda_npearson` is instead re-read from `loss_cfg`
            every call. A caller that mutates `loss_cfg.lambda_pearson` after
            calling this (e.g. to anneal it, as `mich.MICH._shared_step` does
            for `x_phase_loss`) will silently have no effect on the returned
            callable -- only mutating `lambda_npearson` actually changes its
            behaviour.

        Raises:
            ValueError: If `loss_cfg.type` doesn't match a supported name.
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

    class _NoPearsonView:
        """Proxies a loss config but reports `lambda_pearson=0.0`, for building a
        Pearson-free variant of an otherwise-identical loss fn (see `_make_loss_fn`'s
        `lambda_pearson`/`lambda_npearson` handling). All other attributes (`type`,
        `huber_delta`, `lambda_npearson`) pass through to `inner` unchanged."""

        def __init__(self, inner: Any) -> None:
            self._inner = inner

        def __getattr__(self, name: str) -> Any:
            if name == "lambda_pearson":
                return 0.0
            return getattr(self._inner, name)

    @classmethod
    def _make_grid_loss_fn(cls, loss_cfg) -> callable:
        """Like `_make_loss_fn`, but with any Pearson term forced to zero weight.

        Collocation-gathered tensors (`_gather_bold_at_layer`/`_gather_z_hat_at_layer`
        applied to a per-layer collocation batch) have shape [B, n_times, n_space]
        where `dim=1` is *not* one location's trajectory over time -- both the time
        index and the (h, w) location vary independently across it (see
        `_sample_collocation_indices_one_layer`). `_make_loss_fn`'s Pearson term
        assumes `dim=1` is a coherent sequence to correlate pred against true; applied
        here it centers/correlates values from unrelated (t, h, w) points, which is
        not a meaningful signal (unlike the dense/source components, which really do
        gather one fixed location's full trajectory). Use this for any loss fn applied
        to a collocation-batch gather; use `_make_loss_fn` (full Pearson if
        configured) for dense/source-voxel gathers.
        """
        if loss_cfg is None:
            return F.mse_loss
        return cls._make_loss_fn(cls._NoPearsonView(loss_cfg))

    @staticmethod
    def _compute_bold(
        v: torch.Tensor, q: torch.Tensor, acquisition: AcquisitionConstants, V0: float
    ) -> torch.Tensor:
        """BOLD readout from blood volume/deoxyhemoglobin: `V0 * (k1*(1-q) +
        k2*(1-q/v) + k3*(1-v))` (same formula as `mich.data.balloon.get_bold_from_state`,
        with `k1`/`k2`/`k3` from `acquisition` rather than looked up there).

        Warning:
            Divides by `v` elementwise; callers must keep `v` away from 0
            (`_sanitise_states` enforces a 0.1 floor upstream of this in the
            physics-loss path).
        """
        k1, k2, k3 = acquisition.k1, acquisition.k2, acquisition.k3
        return V0 * (k1 * (1 - q) + k2 * (1 - q / v) + k3 * (1 - v))

    @staticmethod
    def _compute_bold_at(
        z_hat: torch.Tensor, idx, acquisition: AcquisitionConstants, V0: float
    ) -> torch.Tensor:
        """`_compute_bold`, gathering v/q from `z_hat` at `idx`'s collocation points
        first. `z_hat`: [B, 7, L, T, H, W]; returns [B, L, n_times, n_space]."""
        v = CollocationMixin._gather_z_hat_at(z_hat, idx, signal="v")
        q = CollocationMixin._gather_z_hat_at(z_hat, idx, signal="q")
        return MICHLossMixin._compute_bold(v, q, acquisition, V0)

    def _source_activity_loss(
        self,
        z_hat: torch.Tensor,  # [B, 7, L, T, H, W]
        source_position: torch.Tensor,  # [B, S, 2]
        source_layer: torch.Tensor,  # [B, S]
        num_sources: torch.Tensor,  # [B]
        eps: float,
    ) -> torch.Tensor:
        """At a source's own (layer, h, w), x should show real dynamics over time (not
        settle to steady state). Penalises var(x) < eps at the labelled source voxel --
        stops the trivial "predict a flat constant" collapse there.
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
        x_src = z_hat[b_idx, x_idx, src_l, :, src_h, src_w]  # [B, S, T]
        x_var = x_src.var(dim=-1)  # [B, S]
        pos_term = F.relu(eps - x_var)
        return (pos_term * mask).sum() / mask.sum().clamp(min=1)

    def _quiescence_consistency_loss(
        self,
        z_hat: torch.Tensor,  # [B, 7, L, T, H, W]
        tau_s: float,
        tau_f: float,
        eps_x: float,
    ) -> torch.Tensor:
        """Self-consistency off-source guard: wherever the model's own s and f sit at
        their resting baseline (|s| < tau_s and |f-1| < tau_f), its own x should also
        be near 0, and penalises it if not.

        Follows directly from the s-equation (see module docstring): if s and ds/dt
        are both ~0 (baseline and unchanging), then x ~= kappa*s + gamma*(f-1), which
        is also ~0 once f is at its baseline of 1. No comparison against a known
        source position or physics constant is needed -- tau_s/tau_f/eps_x are
        generic numerical tolerances, and the gate is evaluated pointwise from z_hat
        at every voxel, layer, and timestep, so it re-applies (densely, over the full
        grid) the same constraint the physics loss only enforces at sparse,
        source-biased collocation points (see `dense_spatial_frac`).
        """
        x_idx, s_idx, f_idx = (
            self._signal_index("x"),
            self._signal_index("s"),
            self._signal_index("f"),
        )
        s = z_hat[:, s_idx]
        f = z_hat[:, f_idx]
        x = z_hat[:, x_idx]

        quiescent = (s.abs() < tau_s) & ((f - 1.0).abs() < tau_f)  # [B, L, T, H, W]
        penalty = F.relu(x.abs() - eps_x)
        return (penalty * quiescent).sum() / quiescent.sum().clamp(min=1)

    def _data_loss(
        self,
        z_hat: torch.Tensor,
        bold_norm: torch.Tensor,
        source_position: torch.Tensor | None = None,
        num_sources: torch.Tensor | None = None,
        source_layer: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """BOLD data loss: sparse grid collocation, plus a heavier-weighted dense
        term at every known source voxel across all of T.

        Predicted BOLD is reconstructed from z_hat's v/q channels via
        `_compute_bold` and PSF-blurred before either component is computed, so
        both are compared against the (already-blurred) `bold_norm`/`true_bold`
        the network is trained against. The collocation term uses
        `_bold_grid_loss_fn` (Pearson-free -- see `_make_grid_loss_fn`) since its
        gather's `dim=1` mixes independently-scattered (t, h, w) points rather than
        one location's trajectory; the dense source term uses the full
        `_bold_loss_fn` (Pearson included if configured), since that gather really is
        one fixed location's trajectory over all of T.

        Args:
            z_hat: [B, 7, L, T, H, W].
            bold_norm: [B, L, T, H, W], the normalised BOLD the model was
                conditioned on; de-normalised via `self.normaliser` (if set)
                to get `true_bold` for comparison.
            source_position, num_sources: See *Source metadata convention* in
                the module docstring -- [B, S, 2] / [B].
            source_layer: [B, S]. If given, each source's dense term compares
                only against its own layer. If None, falls back to comparing
                every layer at that source's (h, w) -- the pre-per-layer
                behaviour, kept for callers without per-layer source info.

        Returns:
            (total, colloc_loss, src_loss): `total = colloc_loss +
            lambda_src * src_loss`; `colloc_loss` and `src_loss` are also
            returned unweighted, for logging.
        """
        L = z_hat.shape[2]
        lc = self.hparams.loss_config
        common_kwargs = dict(
            T=bold_norm.shape[2],
            H=bold_norm.shape[3],
            W=bold_norm.shape[4],
            n_times=lc.n_time,
            n_space=lc.n_space,
            device=z_hat.device,
            dense_spatial_frac=lc.dense_spatial_frac,
            dense_spatial_radius=lc.dense_spatial_radius,
            dense_time_frac=lc.dense_time_frac,
            dense_time_lo=lc.dense_time_lo,
            dense_time_hi=lc.dense_time_hi,
            uniform_time_lo=lc.uniform_time_lo,
        )
        if source_position is not None and source_layer is not None:
            idx_per_layer = self._sample_collocation_indices_per_layer(
                L=L,
                source_position=source_position,
                source_layer=source_layer,
                num_sources=num_sources,
                **common_kwargs,
            )
        else:
            # No per-layer source info -- fall back to one shared draw reused for every
            # layer, matching the old (pre-per-layer) behaviour exactly.
            shared_idx = self._sample_collocation_indices(
                source_position=source_position, num_sources=num_sources, **common_kwargs
            )
            idx_per_layer = [shared_idx] * L

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

        # Collocation loss -- one independent draw per layer; per-layer shape
        # [B, n_times, n_space], where dim=1 is *not* a coherent time series (each
        # entry is an independently-scattered (t, h, w) point -- see
        # `_make_grid_loss_fn`), so this uses the Pearson-free variant.
        colloc_loss = torch.stack(
            [
                self._bold_grid_loss_fn(
                    self._gather_bold_at_layer(pred_bold, idx_per_layer[layer], layer),
                    self._gather_bold_at_layer(true_bold, idx_per_layer[layer], layer),
                )
                for layer in range(L)
            ]
        ).mean()

        # Source voxel loss -- full T, per valid source; Pearson over T (dim=1)
        B = pred_bold.shape[0]
        S = source_position.shape[1]
        b_idx = torch.arange(B, device=pred_bold.device)[:, None].expand(B, S)
        src_h = source_position[..., 0].long()  # [B, S]
        src_w = source_position[..., 1].long()
        mask = torch.arange(S, device=pred_bold.device)[None, :] < num_sources[:, None]  # [B, S]

        if source_layer is not None:
            # Each source's own layer only -- shape [B, S, T].
            src_l = source_layer.clamp(min=0, max=L - 1).long()
            pred_bold_src = pred_bold[b_idx, src_l, :, src_h, src_w]
            true_bold_src = true_bold[b_idx, src_l, :, src_h, src_w]
            T_src = pred_bold_src.shape[-1]
            pred_bold_src = pred_bold_src.reshape(B * S, T_src)[mask.reshape(-1)]  # [M, T]
            true_bold_src = true_bold_src.reshape(B * S, T_src)[mask.reshape(-1)]
            src_loss = self._bold_loss_fn(pred_bold_src, true_bold_src)
        else:
            # No per-layer source info -- fall back to comparing every layer at each
            # source's (h, w), matching the old (pre-per-layer) behaviour exactly.
            pred_bold_src = pred_bold[b_idx, :, :, src_h, src_w]  # [B, S, L, T]
            true_bold_src = true_bold[b_idx, :, :, src_h, src_w]  # [B, S, L, T]
            T_src = pred_bold_src.shape[-1]
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
        """Clamp gathered Heinzle states in place to keep the physics-loss ODE
        residuals (which divide by v and raise v/f to negative/fractional powers)
        finite, before they're used as targets.

        Args:
            states: Signal name -> tensor, values as gathered by
                `_gather_z_hat_at_layer` (any shape). Only keys "f", "v", "q"
                get the positive floor; every other key (e.g. "x", "s",
                "vstar", "qstar") gets the signed clamp.

        Returns:
            The same dict, mutated in place (also returned for chaining).

        Warning:
            NaN/+-inf values are silently replaced (NaN -> 0, +-inf -> +-1e3)
            before clamping. A genuine upstream divergence (e.g. the network
            predicting NaN) will therefore not crash here -- it will instead
            surface downstream as an anomalously large-but-finite physics loss,
            not as an exception.
        """
        for key, value in states.items():
            value = torch.nan_to_num(value, nan=0.0, posinf=1e3, neginf=-1e3)
            if key in ("f", "v", "q"):
                value = torch.clamp(value, min=0.1)
            else:
                value = torch.clamp(value, min=-1e3, max=1e3)
            states[key] = value
        return states

    def _balloon_v_q_dot_targets(
        self,
        f: torch.Tensor,
        v: torch.Tensor,
        q: torch.Tensor,
        order: str,
        need_v: bool = True,
        need_q: bool = True,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Un-normalised (not yet divided by tau) dv/dt, dq/dt targets from the balloon
        ODE's v/q equations (order in {exact, linear, quadratic}). Shared by
        _compute_physics_layer_loss (evaluated on model states) and
        _derivative_supervision_loss (evaluated on ground-truth states) so the two
        formulas can't drift out of sync with each other.

        need_v/need_q let a caller that only supervises one of the two skip computing
        the other; both default to True to preserve the original always-compute-both
        behaviour for _compute_physics_layer_loss.

        Returns:
            (target_vdot, target_qdot); each None if its `need_*` flag is False.

        Raises:
            ValueError: If `order` is not one of "exact", "linear", "quadratic".
        """
        if order not in ("exact", "linear", "quadratic"):
            raise ValueError(
                f"Expected order to be one of `linear`, `quadratic` or `exact`. But received {order}"
            )
        alpha = self._physio("alpha")
        E0 = self._physio("E0")

        target_vdot = None
        if need_v:
            if order == "exact":
                target_vdot = f - v ** (1 / alpha)
            elif order == "linear":
                f_, v_ = f - 1, v - 1
                target_vdot = f_ - v_ / alpha
            elif order == "quadratic":
                f_, v_ = f - 1, v - 1
                target_vdot = f_ - v_ / alpha - (1 - alpha) / (2 * alpha**2) * v_**2

        target_qdot = None
        if need_q:
            if order == "exact":
                target_qdot = f * (1 - (1 - E0) ** (1 / f)) / E0 - q * v ** (1 / alpha - 1)
            elif order == "linear":
                f_, v_, q_ = f - 1, v - 1, q - 1
                log_1mE0 = torch.log(torch.as_tensor(1 - E0, dtype=v.dtype, device=v.device))
                beta_1 = (1 - E0) * log_1mE0 / E0
                target_qdot = (1 + beta_1) * f_ - q_ - (1 / alpha - 1) * v_
            elif order == "quadratic":
                f_, v_, q_ = f - 1, v - 1, q - 1
                log_1mE0 = torch.log(torch.as_tensor(1 - E0, dtype=v.dtype, device=v.device))
                beta_1 = (1 - E0) * log_1mE0 / E0
                beta_2 = beta_1 * log_1mE0 / 2
                target_qdot = (
                    (1 + beta_1) * f_
                    - q_
                    - (1 / alpha - 1) * v_
                    - beta_2 * f_**2
                    - (1 / alpha - 1) * v_ * q_
                    - (1 / 2) * (1 / alpha - 1) * (1 / alpha - 2) * v_**2
                )

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
        """ODE-residual loss for one layer: `self._ode_loss_fn(analytic d/dt, target)`
        per equation, for s/f/v/q (and vstar/qstar in drain mode), at `idx`'s
        collocation points.

        States are gathered and passed through `_sanitise_states` before use (see
        its docstring for the silent NaN/clamp behaviour that implies). `dz_hat_dt`
        slices are converted from t_norm to physical units per the module
        docstring's *Time-derivative units convention* before comparison.

        Args:
            z_hat, dz_hat_dt: [B, 7, L, T, H, W] (dz_hat_dt: 5 channels if not
                `has_drain`).
            idx: This layer's `CollocationBatch` (see `_sample_collocation_indices_per_layer`).
            layer: Fixed layer channel index.
            burn_in: Number of leading timesteps (along the collocation "time"
                axis, i.e. `idx.t`, not the raw T axis) excluded from every loss
                -- early temporal-encoder predictions are unreliable (little
                past context yet); see *Physics loss* in `notebooks/training.md`.
            order: Balloon ODE order, forwarded to `_balloon_v_q_dot_targets`.

        Returns:
            Dict of per-equation scalar losses, keyed "s", "f", "v", "q" (plus
            "vstar", "qstar" if `has_drain`).
        """
        has_drain = z_hat.shape[1] > 5  # vstar/qstar only present in multi-layer mode

        x = self._gather_z_hat_at_layer(z_hat, idx, layer, signal="x")
        s = self._gather_z_hat_at_layer(z_hat, idx, layer, signal="s")
        f = self._gather_z_hat_at_layer(z_hat, idx, layer, signal="f")
        v = self._gather_z_hat_at_layer(z_hat, idx, layer, signal="v")
        q = self._gather_z_hat_at_layer(z_hat, idx, layer, signal="q")

        state_dict = {"x": x, "s": s, "f": f, "v": v, "q": q}
        if has_drain:
            v_star = self._gather_z_hat_at_layer(z_hat, idx, layer, signal="vstar")
            q_star = self._gather_z_hat_at_layer(z_hat, idx, layer, signal="qstar")
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

        _alpha = self._physio("alpha")
        gamma = self._physio("gamma")
        kappa = self._physio("kappa")
        lambda_d = self.hparams.haemo.lambda_d  # not learnable (currently out of scope)
        tau = self._physio("tau")
        tau_d = self.hparams.haemo.tau_d  # not learnable (currently out of scope)
        _E0 = self._physio("E0")

        ds_dt = self._gather_grad_at(dz_hat_dt, layer, idx, signal="s") / t_norm_to_physical
        s_target = x - kappa * s - gamma * (f - 1)
        s_loss = self._ode_loss_fn(ds_dt[:, burn_in:], s_target[:, burn_in:])

        df_dt = self._gather_grad_at(dz_hat_dt, layer, idx, signal="f") / t_norm_to_physical
        f_loss = self._ode_loss_fn(df_dt[:, burn_in:], s[:, burn_in:])

        dv_dt = self._gather_grad_at(dz_hat_dt, layer, idx, signal="v") / t_norm_to_physical
        target_vdot, target_qdot = self._balloon_v_q_dot_targets(f, v, q, order)
        if has_drain and layer > 0:
            vstar_deeper = self._gather_z_hat_at_layer(z_hat, idx, layer - 1, signal="vstar")
            target_vdot = target_vdot + lambda_d * vstar_deeper
        v_loss = self._ode_loss_fn(
            dv_dt[:, burn_in:],
            target_vdot[:, burn_in:] / tau,
        )

        dq_dt = self._gather_grad_at(dz_hat_dt, layer, idx, signal="q") / t_norm_to_physical
        if has_drain and layer > 0:
            qstar_deeper = self._gather_z_hat_at_layer(z_hat, idx, layer - 1, signal="qstar")
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
        anneal_end_step], clamped to start_val before and end_val after.

        Note:
            Reads `self.global_step` (set by `LightningModule`/the trainer) --
            not a pure function of its arguments alone; two calls with
            identical arguments return different results as training
            progresses.

        Returns:
            `end_val` if `anneal_end_step <= anneal_start_step` (a
            zero-or-negative-length window is treated as "already annealed",
            not as an error).
        """
        if anneal_end_step <= anneal_start_step:
            return end_val
        frac = (self.global_step - anneal_start_step) / (anneal_end_step - anneal_start_step)
        frac = min(1.0, max(0.0, frac))
        return start_val + frac * (end_val - start_val)

    def _get_scheduled_lambda(
        self, lambda_target: float, warmup_steps: int, delay_steps: int = 0
    ) -> float:
        """Scale a target loss weight for the current step: 0 before
        `delay_steps`, then linearly ramping to `lambda_target` over the next
        `warmup_steps`, then held at `lambda_target`. With both <= 0, returns
        `lambda_target` unscheduled (always on, from step 0).

        Note:
            Reads `self.global_step`, like `_anneal_between` -- not a pure
            function of its arguments alone.
        """
        if warmup_steps <= 0 and delay_steps <= 0:
            return lambda_target
        if self.global_step < delay_steps:
            return 0.0
        ramp_step = self.global_step - delay_steps
        if warmup_steps <= 0:
            return lambda_target
        return min(1.0, ramp_step / warmup_steps) * lambda_target

    def _physics_loss(
        self,
        z_hat: torch.Tensor,
        dz_hat_dt: torch.Tensor,
        order: str,
        lambda_smooth: float = 0.0,
        source_position: torch.Tensor | None = None,
        source_layer: torch.Tensor | None = None,
        num_sources: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Total ODE-residual loss: `_compute_physics_layer_loss` averaged over
        every equation and every layer, plus an optional temporal-smoothness term.

        Args:
            z_hat, dz_hat_dt: [B, 7, L, T, H, W] (5 channels if not drain mode).
            order: Balloon ODE order ("exact"|"linear"|"quadratic").
            lambda_smooth: Weight for the smoothness term below; <= 0 disables
                it entirely (the term is not computed at all, not computed
                and zero-weighted).
            source_position, source_layer, num_sources: See *Source metadata
                convention* in the module docstring. If `source_position`/
                `source_layer` are None, falls back to one collocation draw
                shared across every layer (pre-per-layer behaviour).

        Returns:
            (total_physics_loss, per_eq): `per_eq` maps each equation name
            ("s","f","v","q"[,"vstar","qstar"]) to its own averaged-over-layers
            scalar, for logging; `total_physics_loss` is their combination
            (equations and layers both averaged, i.e. divided by `n_eq` and by
            `n_layers`) plus `lambda_smooth * smoothness_loss` if enabled.

        Note:
            The smoothness term is a finite difference of `z_hat` over the
            full T axis (`z_hat[..., 1:] - z_hat[..., :-1]`), not restricted to
            collocation points -- unlike every other term here.
        """
        n_layers = z_hat.shape[2]
        lc = self.hparams.loss_config
        common_kwargs = dict(
            T=z_hat.shape[3],
            H=z_hat.shape[4],
            W=z_hat.shape[5],
            n_times=lc.n_time,
            n_space=lc.n_space,
            device=z_hat.device,
            dense_spatial_frac=lc.dense_spatial_frac,
            dense_spatial_radius=lc.dense_spatial_radius,
            dense_time_frac=lc.dense_time_frac,
            dense_time_lo=lc.dense_time_lo,
            dense_time_hi=lc.dense_time_hi,
            uniform_time_lo=lc.uniform_time_lo,
        )
        if source_position is not None and source_layer is not None:
            idx_per_layer = self._sample_collocation_indices_per_layer(
                L=n_layers,
                source_position=source_position,
                source_layer=source_layer,
                num_sources=num_sources,
                **common_kwargs,
            )
        else:
            # No per-layer source info -- fall back to one shared draw reused for every
            # layer, matching the old (pre-per-layer) behaviour exactly.
            shared_idx = self._sample_collocation_indices(
                source_position=source_position, num_sources=num_sources, **common_kwargs
            )
            idx_per_layer = [shared_idx] * n_layers
        has_drain = z_hat.shape[1] > 5
        _eq_keys = ("s", "f", "v", "q", "vstar", "qstar") if has_drain else ("s", "f", "v", "q")
        n_eq = len(_eq_keys)
        tot_physics_loss = torch.tensor(0.0, device=z_hat.device, dtype=torch.float32)
        per_eq = {k: torch.tensor(0.0, device=z_hat.device, dtype=torch.float32) for k in _eq_keys}
        for layer in range(n_layers):
            layer_losses = self._compute_physics_layer_loss(
                z_hat,
                dz_hat_dt,
                idx_per_layer[layer],
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
        """(z_hat signal name, batch dict key) pairs `_supervision_loss` should
        supervise: `_SUPERVISION_KEYS_SINGLE`/`_FULL` depending on whether `z_hat`
        has drain channels, plus `("x", "neural")` if `supervise_x` is set."""
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
        source_position: torch.Tensor,  # [B, S, 2]
        source_layer: torch.Tensor,  # [B, S]
        num_sources: torch.Tensor,  # [B]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """MSE(+Pearson) between predicted and ground-truth latent states (s, f, v, q,
        vstar, qstar), combining two coverage regimes under one term instead of having
        this supervision computed twice by two separate loss functions:

          - a dense component: every valid source, at its own (layer, h, w), across
            every timestep -- guarantees the known source voxel is always fully
            supervised, and (unlike the old _source_supervision_loss this replaces)
            compares it against ground truth only in *its own* layer via
            source_layer, not every layer at that (h, w). Uses the full
            `supervision_loss` config (Pearson included if configured) since this is a
            real, fixed-location trajectory.
          - a grid component: one independent collocation draw per layer, sampled
            across the whole grid, with each layer's dense-near-source share
            restricted to that layer's own sources (see
            _sample_collocation_indices_per_layer) -- this is what reaches
            off-source/weak-spillover voxels, which the dense component never does.
            Uses `_make_grid_loss_fn`'s Pearson-free variant, since each collocation
            gather's `dim=1` mixes independently-scattered (t, h, w) points rather
            than one location's trajectory.

        Combined the same way _data_loss combines its src_loss and colloc_loss:
        weighted by lambda_src so the known source voxel counts more than an
        arbitrary collocation point, without a second, separately-tuned lambda.
        """
        T_latent = batch["s"].shape[2]
        T_min = min(z_hat.shape[3], T_latent)
        L = z_hat.shape[2]
        lc = self.hparams.loss_config
        idx_per_layer = self._sample_collocation_indices_per_layer(
            T=T_min,
            H=z_hat.shape[4],
            W=z_hat.shape[5],
            L=L,
            n_times=lc.n_time,
            n_space=lc.n_space,
            device=z_hat.device,
            source_position=source_position,
            source_layer=source_layer,
            num_sources=num_sources,
            dense_spatial_frac=lc.dense_spatial_frac,
            dense_spatial_radius=lc.dense_spatial_radius,
            dense_time_frac=lc.dense_time_frac,
            dense_time_lo=lc.dense_time_lo,
            dense_time_hi=lc.dense_time_hi,
            uniform_time_lo=lc.uniform_time_lo,
        )

        B = z_hat.shape[0]
        S = source_position.shape[1]
        device = z_hat.device
        b_idx = torch.arange(B, device=device)[:, None].expand(B, S)
        src_h = source_position[..., 0].long()  # [B, S]
        src_w = source_position[..., 1].long()
        src_l = source_layer.clamp(min=0, max=L - 1).long()  # [B, S]
        mask = torch.arange(S, device=device)[None, :] < num_sources[:, None]  # [B, S]

        per_sig: dict[str, torch.Tensor] = {}
        for sig, bk in self._supervision_keys(z_hat):
            true = batch[bk].float()  # [B, L, T_latent, H, W]
            pred = z_hat[:, self._signal_index(sig)].float()  # [B, L, T, H, W]

            # Dense: every valid source, its own layer only, every timestep.
            pred_src = pred[b_idx, src_l, :T_min, src_h, src_w]  # [B, S, T_min]
            true_src = true[b_idx, src_l, :T_min, src_h, src_w]  # [B, S, T_min]
            pred_src = pred_src.reshape(B * S, T_min)[mask.reshape(-1)]  # [M, T_min]
            true_src = true_src.reshape(B * S, T_min)[mask.reshape(-1)]
            dense_loss = self._supervision_loss_fn(pred_src, true_src)

            # Grid: sparse collocation, one independent draw per layer. dim=1 here is
            # not a coherent time series (each entry is an independently-scattered
            # (t, h, w) point -- see `_make_grid_loss_fn`), so this uses the
            # Pearson-free variant.
            grid_loss = torch.stack(
                [
                    self._supervision_grid_loss_fn(
                        self._gather_z_hat_at_layer(
                            z_hat, idx_per_layer[layer], layer, signal=sig
                        ).float(),
                        self._gather_bold_at_layer(true, idx_per_layer[layer], layer).float(),
                    )
                    for layer in range(L)
                ]
            ).mean()

            per_sig[sig] = grid_loss + lc.lambda_src * dense_loss
        total = sum(per_sig.values()) / len(per_sig)
        return total, per_sig

    def _derivative_supervision_loss(
        self,
        dz_hat_dt: torch.Tensor,  # [B, 7, L, T, H, W]
        batch: dict,
        source_position: torch.Tensor,  # [B, S, 2]
        num_sources: torch.Tensor,  # [B]
        source_layer: torch.Tensor | None = None,  # [B, S]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """MSE between the network's analytic dz_hat/dt and the exact Balloon-Windkessel
        ODE derivative evaluated on ground-truth latents, at each valid source voxel
        across all T. When source_layer is given, each source is compared only against
        its own layer, not every layer at that (h, w) (falls back to the old
        every-layer behaviour if source_layer is omitted).

        Compared in per-index units (i.e. per stored sample, not per unit of the model's
        normalised [0, 1] time grid): dz_hat_dt is d(z_hat)/d(t_norm), so it's divided by
        (T_min - 1) to undo that normalisation, matching _compute_physics_layer_loss's
        identical convention.

        Layers > 0 get the same drain-coupling term _compute_physics_layer_loss adds to
        its v/q ODE residual target (+ lambda_d * vstar/qstar from the layer below), built
        here from ground-truth v_star/q_star (batch["v_star"]/["q_star"]) rather than the
        model's own prediction, since every target in this loss is ground-truth-derived.
        vstar/qstar's own analytic targets, (-v_star + v - 1)/tau_d and (-q_star + q - 1)/tau_d,
        mirror _compute_physics_layer_loss's per-layer vstar/qstar targets exactly.
        """
        lc = self.hparams.loss_config
        signals = tuple(getattr(lc, "dzdt_supervision_signals", ("s",)))
        has_drain = dz_hat_dt.shape[1] > 5
        B = dz_hat_dt.shape[0]
        L = dz_hat_dt.shape[2]
        S = source_position.shape[1]
        device = dz_hat_dt.device
        b_idx = torch.arange(B, device=device)[:, None].expand(B, S)
        src_h = source_position[..., 0].long()  # [B, S]
        src_w = source_position[..., 1].long()
        src_l = source_layer.clamp(min=0, max=L - 1).long() if source_layer is not None else None
        mask = torch.arange(S, device=device)[None, :] < num_sources[:, None]  # [B, S]

        s_true = batch["s"].float()
        T_min = min(dz_hat_dt.shape[3], s_true.shape[2])
        s_true = s_true[:, :, :T_min]

        analytic_target: dict[str, torch.Tensor] = {}
        if "s" in signals:
            x_true = batch["neural"].float()[:, :, :T_min]
            f_true = batch["f"].float()[:, :, :T_min]
            kappa = self._physio("kappa")
            gamma = self._physio("gamma")
            analytic_target["s"] = x_true - kappa * s_true - gamma * (f_true - 1.0)
        if "f" in signals:
            analytic_target["f"] = s_true
        if "v" in signals or "q" in signals:
            f_true = batch["f"].float()[:, :, :T_min]
            v_true = batch["v"].float()[:, :, :T_min]
            q_true = batch["q"].float()[:, :, :T_min]
            tau = self._physio("tau")
            target_vdot, target_qdot = self._balloon_v_q_dot_targets(
                f_true, v_true, q_true, lc.order, need_v="v" in signals, need_q="q" in signals
            )
            if has_drain:
                lambda_d = self.hparams.haemo.lambda_d
                if "v" in signals:
                    v_star_true = batch["v_star"].float()[:, :, :T_min]
                    drain_v = torch.zeros_like(target_vdot)
                    drain_v[:, 1:] = lambda_d * v_star_true[:, :-1]
                    target_vdot = target_vdot + drain_v
                if "q" in signals:
                    q_star_true = batch["q_star"].float()[:, :, :T_min]
                    drain_q = torch.zeros_like(target_qdot)
                    drain_q[:, 1:] = lambda_d * q_star_true[:, :-1]
                    target_qdot = target_qdot + drain_q
            if "v" in signals:
                analytic_target["v"] = target_vdot / tau
            if "q" in signals:
                analytic_target["q"] = target_qdot / tau
        if "vstar" in signals or "qstar" in signals:
            if not has_drain:
                raise ValueError(
                    "dzdt_supervision_signals includes vstar/qstar but dz_hat_dt has only "
                    "5 channels -- these signals only exist in multi-layer/drain mode"
                )
            tau_d = self.hparams.haemo.tau_d
            v_true = batch["v"].float()[:, :, :T_min]
            q_true = batch["q"].float()[:, :, :T_min]
            if "vstar" in signals:
                v_star_true = batch["v_star"].float()[:, :, :T_min]
                analytic_target["vstar"] = (-v_star_true + v_true - 1) / tau_d
            if "qstar" in signals:
                q_star_true = batch["q_star"].float()[:, :, :T_min]
                analytic_target["qstar"] = (-q_star_true + q_true - 1) / tau_d

        per_sig: dict[str, torch.Tensor] = {}
        for sig in signals:
            true_dt = analytic_target[sig]  # [B, L, T_min, H, W]
            pred = dz_hat_dt[:, self._signal_index(sig)].float()  # [B, L, T, H, W]

            if src_l is not None:
                # Each source's own layer only -- shape [B, S, T_min].
                pred_src = pred[b_idx, src_l, :T_min, src_h, src_w] / (T_min - 1)
                true_src = true_dt[b_idx, src_l, :T_min, src_h, src_w]
                pred_src = pred_src.reshape(B * S, T_min)[mask.reshape(-1)]  # [M, T_min]
                true_src = true_src.reshape(B * S, T_min)[mask.reshape(-1)]
                per_sig[sig] = self._dzdt_loss_fn(pred_src, true_src)
            else:
                # No per-layer source info -- fall back to comparing every layer at
                # each source's (h, w), matching the old (pre-per-layer) behaviour.
                pred_src = pred[b_idx, :, :T_min, src_h, src_w] / (T_min - 1)  # [B, S, L, T_min]
                true_src = true_dt[b_idx, :, :T_min, src_h, src_w]  # [B, S, L, T_min]
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
        s/f/v/q, by contrast, are judged by _supervision_loss's dense component on their
        full, ordered T-length trajectory at the fixed source voxel every step -- a
        whole-shape comparison, not a cloud of independent point constraints. This loss gives x_hat
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
