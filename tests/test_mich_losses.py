"""Unit tests for MICHLossMixin (src/mich/models/mich_losses.py)."""

from __future__ import annotations

import types

import pytest
import torch
from mich.models.mich_losses import MICHLossMixin
from mich.models.physio import LearnablePhysioMixin


class _LossHost(MICHLossMixin, LearnablePhysioMixin, torch.nn.Module):
    """Minimal concrete object mixing in MICHLossMixin (+ CollocationMixin via it,
    + LearnablePhysioMixin for _physio/_current_acquisition)"""

    def __init__(self, loss_config, haemo, acquisition, V0=0.02, global_step=0, psf_fwhm=None):
        super().__init__()
        self.hparams = types.SimpleNamespace(
            loss_config=loss_config,
            haemo=haemo,
            acquisition=acquisition,
            V0=V0,
            psf_fwhm=psf_fwhm,
        )
        self.global_step = global_step
        self._bold_loss_fn = self._make_loss_fn(getattr(loss_config, "bold_loss", None))
        self._ode_loss_fn = self._make_loss_fn(getattr(loss_config, "ode_loss", None))
        self._supervision_loss_fn = self._make_loss_fn(
            getattr(loss_config, "supervision_loss", None)
        )
        self._dzdt_loss_fn = self._make_loss_fn(getattr(loss_config, "dzdt_loss", None))
        self._x_phase_loss_fn = self._make_loss_fn(getattr(loss_config, "x_phase_loss", None))
        self._setup_learnable_physio(None)
        self._setup_psf()


def _mk_haemo(**overrides):
    base = dict(kappa=0.65, gamma=0.41, alpha=0.32, tau=1.0, lambda_d=0.2, tau_d=1.0)
    base.update(overrides)
    return types.SimpleNamespace(**base)


def _mk_acquisition(**overrides):
    base = dict(k1=0.02, k2=0.38, k3=0.38, E0=0.35)
    base.update(overrides)
    return types.SimpleNamespace(**base)


def _mk_loss_config(**overrides):
    base = dict(
        order="linear",
        n_time=4,
        n_space=4,
        dense_spatial_frac=0.5,
        dense_spatial_radius=2,
        dense_time_frac=0.5,
        dense_time_lo=0.05,
        dense_time_hi=0.55,
        uniform_time_lo=0.05,
        lambda_src=1.0,
        lambda_data=1.0,
        burn_in=1,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


def _mk_host(**loss_cfg_overrides):
    return _LossHost(_mk_loss_config(**loss_cfg_overrides), _mk_haemo(), _mk_acquisition())


# -----------------------------
# _make_loss_fn
# -----------------------------


def test_make_loss_fn_none_defaults_to_mse():
    fn = MICHLossMixin._make_loss_fn(None)
    pred, true = torch.randn(3, 5), torch.randn(3, 5)
    assert torch.isclose(fn(pred, true), torch.nn.functional.mse_loss(pred, true))


def test_make_loss_fn_mse():
    fn = MICHLossMixin._make_loss_fn(types.SimpleNamespace(type="mse"))
    pred, true = torch.randn(3, 5), torch.randn(3, 5)
    assert torch.isclose(fn(pred, true), torch.nn.functional.mse_loss(pred, true))


def test_make_loss_fn_huber_respects_delta():
    fn = MICHLossMixin._make_loss_fn(types.SimpleNamespace(type="huber", huber_delta=0.3))
    pred, true = torch.randn(3, 5), torch.randn(3, 5)
    assert torch.isclose(fn(pred, true), torch.nn.functional.huber_loss(pred, true, delta=0.3))


def test_make_loss_fn_pearson_perfect_correlation_near_zero():
    fn = MICHLossMixin._make_loss_fn(types.SimpleNamespace(type="pearson"))
    x = torch.randn(4, 10)
    assert fn(x, x.clone()).abs() < 1e-5


def test_make_loss_fn_pearson_perfect_anticorrelation_near_two():
    fn = MICHLossMixin._make_loss_fn(types.SimpleNamespace(type="pearson"))
    x = torch.randn(4, 10)
    assert torch.isclose(fn(x, -x), torch.tensor(2.0), atol=1e-4)


def test_make_loss_fn_mse_plus_pearson_combines_both_terms():
    cfg = types.SimpleNamespace(type="mse+pearson", lambda_npearson=2.0, lambda_pearson=0.5)
    fn = MICHLossMixin._make_loss_fn(cfg)
    pred, true = torch.randn(3, 5), torch.randn(3, 5)
    pearson_fn = MICHLossMixin._make_loss_fn(types.SimpleNamespace(type="pearson"))
    expected = 2.0 * torch.nn.functional.mse_loss(pred, true) + 0.5 * pearson_fn(pred, true)
    assert torch.isclose(fn(pred, true), expected, atol=1e-5)


def test_make_loss_fn_huber_plus_pearson_combines_both_terms():
    cfg = types.SimpleNamespace(
        type="huber+pearson", huber_delta=0.2, lambda_npearson=1.5, lambda_pearson=0.7
    )
    fn = MICHLossMixin._make_loss_fn(cfg)
    pred, true = torch.randn(3, 5), torch.randn(3, 5)
    pearson_fn = MICHLossMixin._make_loss_fn(types.SimpleNamespace(type="pearson"))
    expected = 1.5 * torch.nn.functional.huber_loss(pred, true, delta=0.2) + 0.7 * pearson_fn(
        pred, true
    )
    assert torch.isclose(fn(pred, true), expected, atol=1e-5)


def test_make_loss_fn_unknown_type_raises():
    with pytest.raises(ValueError, match="Unrecognised loss type"):
        MICHLossMixin._make_loss_fn(types.SimpleNamespace(type="bogus"))


# -----------------------------
# _compute_bold / _compute_bold_at
# -----------------------------


def test_compute_bold_at_matches_manual_gather_and_formula():
    from mich.models.collocation import CollocationBatch

    B, L, T, H, W = 2, 2, 6, 5, 5
    z_hat = torch.rand(B, 7, L, T, H, W) + 0.5  # keep v away from 0
    idx = CollocationBatch(
        t=torch.randint(0, T, (1, 1, 3, 2)),
        h=torch.randint(0, H, (B, 1, 3, 2)),
        w=torch.randint(0, W, (B, 1, 3, 2)),
    )
    acq = _mk_acquisition()
    from mich.data.balloon import AcquisitionConstants

    ac = AcquisitionConstants(k1=acq.k1, k2=acq.k2, k3=acq.k3)
    bold_at = MICHLossMixin._compute_bold_at(z_hat, idx, ac, V0=0.02)

    v_idx, q_idx = MICHLossMixin._signal_index("v"), MICHLossMixin._signal_index("q")
    v = z_hat[0, v_idx, 0, idx.t[0, 0, 0, 0], idx.h[0, 0, 0, 0], idx.w[0, 0, 0, 0]]
    q = z_hat[0, q_idx, 0, idx.t[0, 0, 0, 0], idx.h[0, 0, 0, 0], idx.w[0, 0, 0, 0]]
    expected = 0.02 * (ac.k1 * (1 - q) + ac.k2 * (1 - q / v) + ac.k3 * (1 - v))
    assert torch.isclose(bold_at[0, 0, 0, 0], expected, atol=1e-6)


# -----------------------------
# _sanitise_states
# -----------------------------


def test_sanitise_states_clamps_positive_and_signed_and_removes_nans():
    host = _mk_host()
    states = {
        "f": torch.tensor([float("nan"), -5.0, 2.0]),
        "v": torch.tensor([float("inf"), 0.0, 1.0]),
        "x": torch.tensor([float("-inf"), 2000.0, -2000.0]),
    }
    out = host._sanitise_states(states)
    assert torch.isfinite(out["f"]).all()
    assert (out["f"] >= 0.1).all()  # positive-clamped group
    assert (out["v"] >= 0.1).all()
    assert torch.isfinite(out["x"]).all()
    assert out["x"].max() <= 1e3 and out["x"].min() >= -1e3  # signed clamp


# -----------------------------
# _balloon_v_q_dot_targets
# -----------------------------


@pytest.mark.parametrize("order", ["exact", "linear", "quadratic"])
def test_balloon_v_q_dot_targets_zero_at_resting_state(order):
    host = _mk_host()
    f = torch.tensor([1.0])
    v = torch.tensor([1.0])
    q = torch.tensor([1.0])
    vdot, qdot = host._balloon_v_q_dot_targets(f, v, q, order)
    assert torch.isclose(vdot, torch.zeros(1), atol=1e-6)
    assert torch.isclose(qdot, torch.zeros(1), atol=1e-6)


def test_balloon_v_q_dot_targets_linear_matches_exact_for_small_perturbation():
    host = _mk_host()
    f = torch.tensor([1.01])
    v = torch.tensor([0.99])
    q = torch.tensor([1.02])
    vdot_exact, qdot_exact = host._balloon_v_q_dot_targets(f, v, q, "exact")
    vdot_lin, qdot_lin = host._balloon_v_q_dot_targets(f, v, q, "linear")
    assert torch.isclose(vdot_exact, vdot_lin, atol=1e-3)
    assert torch.isclose(qdot_exact, qdot_lin, atol=1e-3)


def test_balloon_v_q_dot_targets_invalid_order_raises():
    host = _mk_host()
    with pytest.raises(ValueError, match="Expected order"):
        host._balloon_v_q_dot_targets(
            torch.tensor([1.0]), torch.tensor([1.0]), torch.tensor([1.0]), "bogus"
        )


# -----------------------------
# _compute_physics_layer_loss
# -----------------------------


def _mk_zhat_dzdt(B, n_sig, L, T, H, W, requires_grad=False):
    z_hat = torch.rand(B, n_sig, L, T, H, W) + 0.5
    dz_hat_dt = torch.randn(B, n_sig, L, T, H, W) * 0.01
    if requires_grad:
        z_hat.requires_grad_(True)
        dz_hat_dt.requires_grad_(True)
    return z_hat, dz_hat_dt


def test_compute_physics_layer_loss_single_layer_no_drain_keys():
    host = _mk_host()
    from mich.models.collocation import CollocationBatch

    B, L, T, H, W = 2, 1, 8, 4, 4
    z_hat, dz_hat_dt = _mk_zhat_dzdt(B, 5, L, T, H, W)  # <=5 channels -> has_drain=False
    idx = CollocationBatch(
        t=torch.randint(0, T, (1, 1, 3, 2)),
        h=torch.randint(0, H, (B, 1, 3, 2)),
        w=torch.randint(0, W, (B, 1, 3, 2)),
    )
    losses = host._compute_physics_layer_loss(
        z_hat, dz_hat_dt, idx, layer=0, burn_in=0, order="linear"
    )
    assert set(losses.keys()) == {"s", "f", "v", "q"}
    for v in losses.values():
        assert torch.isfinite(v)


def test_compute_physics_layer_loss_multilayer_with_drain_keys_and_coupling():
    host = _mk_host()
    from mich.models.collocation import CollocationBatch

    B, L, T, H, W = 2, 2, 8, 4, 4
    z_hat, dz_hat_dt = _mk_zhat_dzdt(B, 7, L, T, H, W)  # 7 channels -> has_drain=True
    idx = CollocationBatch(
        t=torch.randint(0, T, (1, 1, 3, 2)),
        h=torch.randint(0, H, (B, 1, 3, 2)),
        w=torch.randint(0, W, (B, 1, 3, 2)),
    )
    losses_layer0 = host._compute_physics_layer_loss(
        z_hat, dz_hat_dt, idx, layer=0, burn_in=0, order="linear"
    )
    losses_layer1 = host._compute_physics_layer_loss(
        z_hat, dz_hat_dt, idx, layer=1, burn_in=0, order="linear"
    )
    assert set(losses_layer0.keys()) == {"s", "f", "v", "q", "vstar", "qstar"}
    assert set(losses_layer1.keys()) == {"s", "f", "v", "q", "vstar", "qstar"}
    for v in {**losses_layer0, **losses_layer1}.values():
        assert torch.isfinite(v)


def test_compute_physics_layer_loss_differentiable():
    host = _mk_host()
    from mich.models.collocation import CollocationBatch

    B, L, T, H, W = 1, 1, 8, 4, 4
    z_hat, dz_hat_dt = _mk_zhat_dzdt(B, 5, L, T, H, W, requires_grad=True)
    idx = CollocationBatch(
        t=torch.randint(0, T, (1, 1, 3, 2)),
        h=torch.randint(0, H, (B, 1, 3, 2)),
        w=torch.randint(0, W, (B, 1, 3, 2)),
    )
    losses = host._compute_physics_layer_loss(
        z_hat, dz_hat_dt, idx, layer=0, burn_in=0, order="exact"
    )
    total = sum(losses.values())
    total.backward()
    assert z_hat.grad is not None and torch.isfinite(z_hat.grad).all()
    assert dz_hat_dt.grad is not None and torch.isfinite(dz_hat_dt.grad).all()


# -----------------------------
# _anneal_between / _get_scheduled_lambda
# -----------------------------


def test_anneal_between_no_op_when_end_not_after_start():
    host = _mk_host()
    host.global_step = 500
    assert host._anneal_between(0.0, 1.0, anneal_start_step=10, anneal_end_step=10) == 1.0
    assert host._anneal_between(0.0, 1.0, anneal_start_step=10, anneal_end_step=5) == 1.0


def test_anneal_between_linear_interpolation_and_clamping():
    host = _mk_host()
    host.global_step = 0
    assert host._anneal_between(
        0.0, 10.0, anneal_start_step=0, anneal_end_step=100
    ) == pytest.approx(0.0)
    host.global_step = 50
    assert host._anneal_between(
        0.0, 10.0, anneal_start_step=0, anneal_end_step=100
    ) == pytest.approx(5.0)
    host.global_step = 200  # past the end -> clamped to end_val
    assert host._anneal_between(
        0.0, 10.0, anneal_start_step=0, anneal_end_step=100
    ) == pytest.approx(10.0)
    host.global_step = -10  # before start -> clamped to start_val
    assert host._anneal_between(
        0.0, 10.0, anneal_start_step=0, anneal_end_step=100
    ) == pytest.approx(0.0)


def test_get_scheduled_lambda_no_warmup_no_delay_returns_target():
    host = _mk_host()
    host.global_step = 5
    assert host._get_scheduled_lambda(2.0, warmup_steps=0, delay_steps=0) == 2.0


def test_get_scheduled_lambda_before_delay_is_zero():
    host = _mk_host()
    host.global_step = 3
    assert host._get_scheduled_lambda(2.0, warmup_steps=10, delay_steps=10) == 0.0


def test_get_scheduled_lambda_ramps_linearly_after_delay():
    host = _mk_host()
    host.global_step = 15  # 5 steps past delay=10, warmup=10 -> 0.5 ramp
    assert host._get_scheduled_lambda(2.0, warmup_steps=10, delay_steps=10) == pytest.approx(1.0)


def test_get_scheduled_lambda_full_target_after_warmup_completes():
    host = _mk_host()
    host.global_step = 1000
    assert host._get_scheduled_lambda(2.0, warmup_steps=10, delay_steps=10) == 2.0


def test_get_scheduled_lambda_delay_only_no_warmup_returns_target_once_passed():
    host = _mk_host()
    host.global_step = 20
    assert host._get_scheduled_lambda(3.0, warmup_steps=0, delay_steps=10) == 3.0


# -----------------------------
# PSF setup / blur
# -----------------------------


def test_setup_psf_none_is_noop():
    host = _mk_host()  # psf_fwhm=None by default
    bold = torch.randn(2, 2, 5, 6, 6)
    assert torch.equal(host._apply_psf_blur(bold), bold)


def test_apply_psf_blur_zero_fwhm_layer_is_near_identity():
    host = _LossHost(_mk_loss_config(), _mk_haemo(), _mk_acquisition(), psf_fwhm=[0.0, 0.0])
    bold = torch.randn(2, 2, 5, 6, 6)
    blurred = host._apply_psf_blur(bold)
    assert torch.allclose(blurred, bold, atol=1e-5)


def test_apply_psf_blur_positive_fwhm_actually_blurs_and_is_per_layer():
    host = _LossHost(_mk_loss_config(), _mk_haemo(), _mk_acquisition(), psf_fwhm=[2.0, 0.0])
    bold = torch.zeros(1, 2, 3, 9, 9)
    bold[:, :, :, 4, 4] = 1.0  # impulse at center of both layers
    blurred = host._apply_psf_blur(bold)
    # Layer 0 (fwhm=2.0) should spread the impulse to neighbours; layer 1 (fwhm=0) should not.
    assert blurred[0, 0, 0, 4, 4] < 1.0
    assert blurred[0, 0, 0, 3, 4] > 0.0
    assert torch.allclose(blurred[0, 1], bold[0, 1], atol=1e-5)


# -----------------------------
# _antisteady_loss
# -----------------------------


def test_antisteady_loss_penalizes_flat_source_and_noisy_other_layers():
    host = _mk_host(antisteady_epsilon=0.01)
    B, L, T, H, W = 1, 2, 20, 6, 6
    z_hat = torch.zeros(B, 7, L, T, H, W)
    x_idx = host._signal_index("x")
    # Source at (h=2,w=3) in layer 0: flat over time -> should be penalized (var ~ 0 < eps).
    z_hat[:, x_idx, 0, :, 2, 3] = 1.0
    # Same (h,w) in the OTHER layer (layer 1): noisy -> should be penalized (var > eps_neg).
    z_hat[:, x_idx, 1, :, 2, 3] = torch.randn(T)

    source_position = torch.tensor([[[2, 3]]])
    source_layer = torch.tensor([[0]])
    num_sources = torch.tensor([1])

    loss = host._antisteady_loss(z_hat, source_position, source_layer, num_sources)
    assert torch.isfinite(loss)
    assert loss > 0.0


def test_antisteady_loss_zero_when_source_dynamic_and_others_flat():
    host = _mk_host(antisteady_epsilon=0.01)
    B, L, T, H, W = 1, 2, 20, 6, 6
    z_hat = torch.zeros(B, 7, L, T, H, W)
    x_idx = host._signal_index("x")
    z_hat[:, x_idx, 0, :, 2, 3] = torch.linspace(0, 1, T)  # real dynamics, var > eps
    # other layer at same voxel stays exactly flat (0) -> no penalty there either

    source_position = torch.tensor([[[2, 3]]])
    source_layer = torch.tensor([[0]])
    num_sources = torch.tensor([1])

    loss = host._antisteady_loss(z_hat, source_position, source_layer, num_sources)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_antisteady_loss_masks_padded_sources():
    host = _mk_host(antisteady_epsilon=0.01)
    B, L, T, H, W = 1, 2, 20, 6, 6
    z_hat = torch.zeros(B, 7, L, T, H, W)
    x_idx = host._signal_index("x")
    z_hat[:, x_idx, 0, :, 2, 3] = torch.linspace(0, 1, T)

    # Slot 1 is padding (num_sources=1): even though its position looks flat, it must
    # not contribute -- verify by comparing against a variant with garbage in slot 1.
    source_position = torch.tensor([[[2, 3], [0, 0]]])
    source_layer = torch.tensor([[0, 0]])
    num_sources = torch.tensor([1])
    loss_masked = host._antisteady_loss(z_hat, source_position, source_layer, num_sources)

    source_position2 = torch.tensor([[[2, 3], [5, 5]]])  # different padding content
    loss_masked2 = host._antisteady_loss(z_hat, source_position2, source_layer, num_sources)
    assert torch.isclose(loss_masked, loss_masked2, atol=1e-6)


# -----------------------------
# _physics_loss (end-to-end through the mixin, no HeinzleNet)
# -----------------------------


def test_physics_loss_single_layer_no_smooth():
    host = _mk_host(lambda_smooth=0.0)
    B, L, T, H, W = 2, 1, 10, 5, 5
    z_hat, dz_hat_dt = _mk_zhat_dzdt(B, 5, L, T, H, W)
    source_position = torch.randint(0, 5, (B, 1, 2))
    num_sources = torch.ones(B, dtype=torch.long)
    loss, per_eq = host._physics_loss(
        z_hat,
        dz_hat_dt,
        order="linear",
        lambda_smooth=0.0,
        source_position=source_position,
        num_sources=num_sources,
    )
    assert torch.isfinite(loss)
    assert set(per_eq.keys()) == {"s", "f", "v", "q"}


def test_physics_loss_adds_smoothness_term_when_lambda_smooth_positive():
    host = _mk_host()
    B, L, T, H, W = 2, 1, 10, 5, 5
    torch.manual_seed(0)
    z_hat, dz_hat_dt = _mk_zhat_dzdt(B, 5, L, T, H, W)
    source_position = torch.randint(0, 5, (B, 1, 2))
    num_sources = torch.ones(B, dtype=torch.long)

    loss_no_smooth, _ = host._physics_loss(
        z_hat,
        dz_hat_dt,
        order="linear",
        lambda_smooth=0.0,
        source_position=source_position,
        num_sources=num_sources,
    )
    torch.manual_seed(0)
    loss_smooth, _ = host._physics_loss(
        z_hat,
        dz_hat_dt,
        order="linear",
        lambda_smooth=100.0,
        source_position=source_position,
        num_sources=num_sources,
    )
    assert loss_smooth > loss_no_smooth


def test_physics_loss_multilayer_has_drain_per_eq_keys():
    host = _mk_host()
    B, L, T, H, W = 2, 2, 10, 5, 5
    z_hat, dz_hat_dt = _mk_zhat_dzdt(B, 7, L, T, H, W)
    source_position = torch.randint(0, 5, (B, 1, 2))
    num_sources = torch.ones(B, dtype=torch.long)
    loss, per_eq = host._physics_loss(
        z_hat,
        dz_hat_dt,
        order="linear",
        lambda_smooth=0.0,
        source_position=source_position,
        num_sources=num_sources,
    )
    assert torch.isfinite(loss)
    assert set(per_eq.keys()) == {"s", "f", "v", "q", "vstar", "qstar"}


# -----------------------------
# _supervision_keys
# -----------------------------


def test_supervision_keys_single_vs_full_by_channel_count():
    host = _mk_host()
    z_hat_5 = torch.zeros(1, 5, 1, 1, 1, 1)
    z_hat_7 = torch.zeros(1, 7, 1, 1, 1, 1)
    assert host._supervision_keys(z_hat_5) == MICHLossMixin._SUPERVISION_KEYS_SINGLE
    assert host._supervision_keys(z_hat_7) == MICHLossMixin._SUPERVISION_KEYS_FULL


def test_supervision_keys_prepends_x_when_supervise_x_true():
    host = _mk_host(supervise_x=True)
    z_hat_5 = torch.zeros(1, 5, 1, 1, 1, 1)
    keys = host._supervision_keys(z_hat_5)
    assert keys[0] == ("x", "neural")
    assert keys[1:] == MICHLossMixin._SUPERVISION_KEYS_SINGLE


# -----------------------------
# _supervision_loss / _source_supervision_loss / _derivative_supervision_loss / _x_phase_loss
# -----------------------------


def _mk_supervision_batch(B, L, T, H, W, S=1):
    batch = {}
    for key in ("s", "f", "v", "q", "v_star", "q_star", "neural"):
        batch[key] = torch.randn(B, L, T, H, W)
    return batch


def test_supervision_loss_perfect_match_gives_zero():
    host = _mk_host(supervision_loss=None)
    B, L, T, H, W = 2, 1, 10, 5, 5
    batch = _mk_supervision_batch(B, L, T, H, W)
    z_hat = torch.zeros(B, 7, L, T, H, W)
    for sig, bk in MICHLossMixin._SUPERVISION_KEYS_FULL:
        z_hat[:, host._signal_index(sig)] = batch[bk]
    source_position = torch.randint(0, 5, (B, 1, 2))
    num_sources = torch.ones(B, dtype=torch.long)
    total, per_sig = host._supervision_loss(z_hat, batch, source_position, num_sources)
    assert torch.isclose(total, torch.tensor(0.0), atol=1e-6)
    assert set(per_sig.keys()) == {"s", "f", "v", "q", "vstar", "qstar"}


def test_supervision_loss_nonzero_for_mismatched_signals():
    host = _mk_host()
    B, L, T, H, W = 2, 1, 10, 5, 5
    batch = _mk_supervision_batch(B, L, T, H, W)
    z_hat = torch.randn(B, 7, L, T, H, W)  # unrelated to batch
    source_position = torch.randint(0, 5, (B, 1, 2))
    num_sources = torch.ones(B, dtype=torch.long)
    total, _ = host._supervision_loss(z_hat, batch, source_position, num_sources)
    assert total > 0.0


def test_supervision_loss_handles_t_min_mismatch():
    host = _mk_host()
    B, L, H, W = 2, 1, 5, 5
    z_hat = torch.randn(B, 7, L, 12, H, W)  # T=12
    batch = _mk_supervision_batch(B, L, 8, H, W)  # T_latent=8 < z_hat's T
    source_position = torch.randint(0, 5, (B, 1, 2))
    num_sources = torch.ones(B, dtype=torch.long)
    total, _ = host._supervision_loss(z_hat, batch, source_position, num_sources)
    assert torch.isfinite(total)


def test_source_supervision_loss_perfect_match_gives_zero():
    host = _mk_host()
    B, L, T, H, W = 2, 1, 10, 5, 5
    batch = _mk_supervision_batch(B, L, T, H, W)
    z_hat = torch.zeros(B, 7, L, T, H, W)
    for sig, bk in MICHLossMixin._SUPERVISION_KEYS_FULL:
        z_hat[:, host._signal_index(sig)] = batch[bk]
    source_position = torch.randint(0, 5, (B, 1, 2))
    num_sources = torch.ones(B, dtype=torch.long)
    total, per_sig = host._source_supervision_loss(z_hat, batch, source_position, num_sources)
    assert torch.isclose(total, torch.tensor(0.0), atol=1e-6)


def test_source_supervision_loss_masks_padded_sources_and_batch_entries():
    host = _mk_host()
    B, S, L, T, H, W = 2, 2, 1, 10, 5, 5
    batch = _mk_supervision_batch(B, L, T, H, W)
    z_hat = torch.zeros(B, 7, L, T, H, W)
    for sig, bk in MICHLossMixin._SUPERVISION_KEYS_FULL:
        z_hat[:, host._signal_index(sig)] = batch[bk]
    source_position = torch.randint(0, 5, (B, S, 2))
    num_sources = torch.tensor([1, 0])  # sample 1 has zero valid sources
    total, _ = host._source_supervision_loss(z_hat, batch, source_position, num_sources)
    assert torch.isclose(total, torch.tensor(0.0), atol=1e-6)


def test_derivative_supervision_loss_matches_analytic_target_when_consistent():
    host = _mk_host(order="linear")
    B, L, T, H, W = 2, 1, 10, 5, 5
    kappa, gamma, tau = 0.65, 0.41, 1.0
    x_true = torch.randn(B, L, T, H, W)
    s_true = torch.randn(B, L, T, H, W)
    f_true = torch.rand(B, L, T, H, W) + 0.5
    v_true = torch.rand(B, L, T, H, W) + 0.5
    q_true = torch.rand(B, L, T, H, W) + 0.5
    batch = {"neural": x_true, "s": s_true, "f": f_true, "v": v_true, "q": q_true}

    vdot, qdot = host._balloon_v_q_dot_targets(f_true, v_true, q_true, "linear")
    analytic = {
        "s": x_true - kappa * s_true - gamma * (f_true - 1.0),
        "f": s_true,
        "v": vdot / tau,
        "q": qdot / tau,
    }
    dz_hat_dt = torch.zeros(B, 7, L, T, H, W)
    t_norm_to_physical = T - 1
    for sig in ("s", "f", "v", "q"):
        dz_hat_dt[:, host._signal_index(sig)] = analytic[sig] * t_norm_to_physical

    source_position = torch.randint(0, 5, (B, 1, 2))
    num_sources = torch.ones(B, dtype=torch.long)
    total, per_sig = host._derivative_supervision_loss(
        dz_hat_dt, batch, source_position, num_sources
    )
    assert torch.isclose(total, torch.tensor(0.0), atol=1e-4)
    assert set(per_sig.keys()) == {"s"}  # default dzdt_supervision_signals=("s",)


def test_derivative_supervision_loss_respects_custom_signal_list():
    host = _mk_host(order="linear", dzdt_supervision_signals=("s", "f"))
    B, L, T, H, W = 2, 1, 10, 5, 5
    batch = {
        "neural": torch.randn(B, L, T, H, W),
        "s": torch.randn(B, L, T, H, W),
        "f": torch.rand(B, L, T, H, W) + 0.5,
        "v": torch.rand(B, L, T, H, W) + 0.5,
        "q": torch.rand(B, L, T, H, W) + 0.5,
    }
    dz_hat_dt = torch.randn(B, 7, L, T, H, W)
    source_position = torch.randint(0, 5, (B, 1, 2))
    num_sources = torch.ones(B, dtype=torch.long)
    _, per_sig = host._derivative_supervision_loss(dz_hat_dt, batch, source_position, num_sources)
    assert set(per_sig.keys()) == {"s", "f"}


def test_x_phase_loss_zero_when_x_matches_its_reconstruction():
    host = _mk_host(burn_in=0)
    B, L, T, H, W = 2, 1, 10, 5, 5
    kappa, gamma = 0.65, 0.41
    s_hat = torch.randn(B, 7, L, T, H, W)
    f_hat = torch.rand(B, 7, L, T, H, W) + 0.5
    z_hat = torch.zeros(B, 7, L, T, H, W)
    z_hat[:, host._signal_index("s")] = s_hat[:, host._signal_index("s")]
    z_hat[:, host._signal_index("f")] = f_hat[:, host._signal_index("f")]

    dz_hat_dt = torch.zeros(B, 7, L, T, H, W)
    Dp_s_phys = torch.randn(B, L, T, H, W) * 0.01
    dz_hat_dt[:, host._signal_index("s")] = Dp_s_phys * (T - 1)

    x_rhs = (
        Dp_s_phys
        + kappa * z_hat[:, host._signal_index("s")]
        + gamma * (z_hat[:, host._signal_index("f")] - 1.0)
    )
    z_hat[:, host._signal_index("x")] = x_rhs

    source_position = torch.randint(0, 5, (B, 1, 2))
    num_sources = torch.ones(B, dtype=torch.long)
    loss = host._x_phase_loss(z_hat, dz_hat_dt, source_position, num_sources)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-4)


def test_x_phase_loss_positive_when_mismatched():
    host = _mk_host(burn_in=0)
    B, L, T, H, W = 2, 1, 10, 5, 5
    z_hat = torch.randn(B, 7, L, T, H, W)
    dz_hat_dt = torch.randn(B, 7, L, T, H, W)
    source_position = torch.randint(0, 5, (B, 1, 2))
    num_sources = torch.ones(B, dtype=torch.long)
    loss = host._x_phase_loss(z_hat, dz_hat_dt, source_position, num_sources)
    assert loss > 0.0
