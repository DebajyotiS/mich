"""Unit tests for MICHLossMixin (src/mich/models/mich_losses.py)."""

from __future__ import annotations

import types
from unittest.mock import patch

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
        self._bold_grid_loss_fn = self._make_grid_loss_fn(getattr(loss_config, "bold_loss", None))
        self._ode_loss_fn = self._make_loss_fn(getattr(loss_config, "ode_loss", None))
        self._supervision_loss_fn = self._make_loss_fn(
            getattr(loss_config, "supervision_loss", None)
        )
        self._supervision_grid_loss_fn = self._make_grid_loss_fn(
            getattr(loss_config, "supervision_loss", None)
        )
        self._dzdt_loss_fn = self._make_loss_fn(getattr(loss_config, "dzdt_loss", None))
        self._x_phase_loss_fn = self._make_loss_fn(getattr(loss_config, "x_phase_loss", None))
        self._setup_learnable_physio(None)
        self._setup_psf()


def _mk_haemo(**overrides):
    # Every constant here is deliberately distinct and non-unit. tau/tau_d of 1.0
    # make every `/ tau` and `/ tau_d` in the ODE targets a no-op, and equal
    # kappa/gamma or k2/k3 make swapping them undetectable.
    base = dict(kappa=0.65, gamma=0.41, alpha=0.32, tau=1.7, lambda_d=0.2, tau_d=2.3)
    base.update(overrides)
    return types.SimpleNamespace(**base)


def _mk_acquisition(**overrides):
    base = dict(k1=0.02, k2=0.38, k3=0.52, E0=0.35)
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
        lambda_src=1.6,
        lambda_data=1.3,
        burn_in=1,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


def _mk_host(global_step=0, **loss_cfg_overrides):
    return _LossHost(
        _mk_loss_config(**loss_cfg_overrides),
        _mk_haemo(),
        _mk_acquisition(),
        global_step=global_step,
    )


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


def test_make_loss_fn_mse_plus_pearson_reads_lambda_pearson_at_call_time():
    """Regression test: lambda_pearson must be re-read on every call, like
    lambda_npearson already is, so annealing it after _make_loss_fn was built
    (e.g. MICH._shared_step's x_phase_loss annealing) actually takes effect."""
    cfg = types.SimpleNamespace(type="mse+pearson", lambda_npearson=1.0, lambda_pearson=0.5)
    fn = MICHLossMixin._make_loss_fn(cfg)
    pred, true = torch.randn(3, 5), torch.randn(3, 5)
    pearson_fn = MICHLossMixin._make_loss_fn(types.SimpleNamespace(type="pearson"))

    before = fn(pred, true)
    expected_before = 1.0 * torch.nn.functional.mse_loss(pred, true) + 0.5 * pearson_fn(pred, true)
    assert torch.isclose(before, expected_before, atol=1e-5)

    cfg.lambda_pearson = 5.0
    after = fn(pred, true)
    expected_after = 1.0 * torch.nn.functional.mse_loss(pred, true) + 5.0 * pearson_fn(pred, true)
    assert torch.isclose(after, expected_after, atol=1e-5)
    assert not torch.isclose(after, before, atol=1e-5)


def test_make_loss_fn_huber_plus_pearson_reads_lambda_pearson_at_call_time():
    cfg = types.SimpleNamespace(
        type="huber+pearson", huber_delta=0.2, lambda_npearson=1.0, lambda_pearson=0.5
    )
    fn = MICHLossMixin._make_loss_fn(cfg)
    pred, true = torch.randn(3, 5), torch.randn(3, 5)
    pearson_fn = MICHLossMixin._make_loss_fn(types.SimpleNamespace(type="pearson"))

    cfg.lambda_pearson = 3.0
    result = fn(pred, true)
    expected = 1.0 * torch.nn.functional.huber_loss(pred, true, delta=0.2) + 3.0 * pearson_fn(
        pred, true
    )
    assert torch.isclose(result, expected, atol=1e-5)


def test_make_loss_fn_unknown_type_raises():
    with pytest.raises(ValueError, match="Unrecognised loss type"):
        MICHLossMixin._make_loss_fn(types.SimpleNamespace(type="bogus"))


# -----------------------------
# _make_grid_loss_fn
# -----------------------------


def test_make_grid_loss_fn_none_defaults_to_mse():
    fn = MICHLossMixin._make_grid_loss_fn(None)
    assert fn is torch.nn.functional.mse_loss


def test_make_grid_loss_fn_mse_plus_pearson_drops_pearson_term():
    cfg = types.SimpleNamespace(type="mse+pearson", lambda_npearson=2.0, lambda_pearson=0.5)
    fn = MICHLossMixin._make_grid_loss_fn(cfg)
    pred, true = torch.randn(3, 5), torch.randn(3, 5)
    expected = 2.0 * torch.nn.functional.mse_loss(pred, true)
    assert torch.isclose(fn(pred, true), expected, atol=1e-5)


def test_make_grid_loss_fn_huber_plus_pearson_drops_pearson_term():
    cfg = types.SimpleNamespace(
        type="huber+pearson", huber_delta=0.2, lambda_npearson=1.5, lambda_pearson=0.7
    )
    fn = MICHLossMixin._make_grid_loss_fn(cfg)
    pred, true = torch.randn(3, 5), torch.randn(3, 5)
    expected = 1.5 * torch.nn.functional.huber_loss(pred, true, delta=0.2)
    assert torch.isclose(fn(pred, true), expected, atol=1e-5)


def test_make_grid_loss_fn_plain_mse_unaffected():
    fn = MICHLossMixin._make_grid_loss_fn(types.SimpleNamespace(type="mse"))
    assert fn is torch.nn.functional.mse_loss


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
# _source_activity_loss
# -----------------------------


def test_source_activity_loss_penalizes_flat_source():
    host = _mk_host()
    B, L, T, H, W = 1, 2, 20, 6, 6
    z_hat = torch.zeros(B, 7, L, T, H, W)
    x_idx = host._signal_index("x")
    # Source at (h=2,w=3) in layer 0: flat over time -> should be penalized (var ~ 0 < eps).
    z_hat[:, x_idx, 0, :, 2, 3] = 1.0

    source_position = torch.tensor([[[2, 3]]])
    source_layer = torch.tensor([[0]])
    num_sources = torch.tensor([1])

    loss = host._source_activity_loss(z_hat, source_position, source_layer, num_sources, eps=0.01)
    assert torch.isfinite(loss)
    assert loss > 0.0


def test_source_activity_loss_zero_when_source_dynamic():
    host = _mk_host()
    B, L, T, H, W = 1, 2, 20, 6, 6
    z_hat = torch.zeros(B, 7, L, T, H, W)
    x_idx = host._signal_index("x")
    z_hat[:, x_idx, 0, :, 2, 3] = torch.linspace(0, 1, T)  # real dynamics, var > eps

    source_position = torch.tensor([[[2, 3]]])
    source_layer = torch.tensor([[0]])
    num_sources = torch.tensor([1])

    loss = host._source_activity_loss(z_hat, source_position, source_layer, num_sources, eps=0.01)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_source_activity_loss_masks_padded_sources():
    host = _mk_host()
    B, L, T, H, W = 1, 2, 20, 6, 6
    z_hat = torch.zeros(B, 7, L, T, H, W)
    x_idx = host._signal_index("x")
    z_hat[:, x_idx, 0, :, 2, 3] = torch.linspace(0, 1, T)

    # Slot 1 is padding (num_sources=1): even though its position looks flat, it must
    # not contribute -- verify by comparing against a variant with garbage in slot 1.
    source_position = torch.tensor([[[2, 3], [0, 0]]])
    source_layer = torch.tensor([[0, 0]])
    num_sources = torch.tensor([1])
    loss_masked = host._source_activity_loss(
        z_hat, source_position, source_layer, num_sources, eps=0.01
    )

    source_position2 = torch.tensor([[[2, 3], [5, 5]]])  # different padding content
    loss_masked2 = host._source_activity_loss(
        z_hat, source_position2, source_layer, num_sources, eps=0.01
    )
    assert torch.isclose(loss_masked, loss_masked2, atol=1e-6)


# -----------------------------
# _quiescence_consistency_loss
# -----------------------------


def test_quiescence_consistency_loss_penalizes_hallucination_at_quiescent_voxel():
    # Where s and f are jointly at baseline (s~=0, f~=1), x should be ~0 too (the
    # s-equation ODE residual forces this). A voxel that's quiescent per s/f but has
    # noisy x is an internal inconsistency and must be penalized.
    host = _mk_host()
    B, L, T, H, W = 1, 1, 20, 4, 4
    f_idx, x_idx = host._signal_index("f"), host._signal_index("x")
    z_hat = torch.zeros(B, 7, L, T, H, W)
    z_hat[:, f_idx] = 1.0  # baseline for f is 1, not 0 -- s already defaults to 0 (baseline)
    z_hat[:, x_idx, 0, :, 0, 0] = torch.randn(T) * 10  # quiescent voxel, hallucinated x

    loss = host._quiescence_consistency_loss(z_hat, tau_s=0.05, tau_f=0.05, eps_x=0.05)
    assert loss > 0.0


def test_quiescence_consistency_loss_zero_when_all_flat():
    host = _mk_host()
    B, L, T, H, W = 1, 1, 20, 4, 4
    f_idx = host._signal_index("f")
    z_hat = torch.zeros(B, 7, L, T, H, W)
    z_hat[:, f_idx] = 1.0  # s=0, f=1, x=0 everywhere -- fully self-consistent quiescence

    loss = host._quiescence_consistency_loss(z_hat, tau_s=0.05, tau_f=0.05, eps_x=0.05)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_quiescence_consistency_loss_exempts_non_quiescent_voxel_regardless_of_x():
    # A voxel where s (or f) is NOT at baseline is exempt from this term entirely, no
    # matter what x does there -- that's a real source/diffusion signature, not
    # hallucination, and is source_activity_loss/physics_loss's job to judge, not this
    # term's.
    host = _mk_host()
    B, L, T, H, W = 1, 1, 20, 4, 4
    s_idx, f_idx, x_idx = host._signal_index("s"), host._signal_index("f"), host._signal_index("x")
    z_hat = torch.zeros(B, 7, L, T, H, W)
    z_hat[:, f_idx] = 1.0
    z_hat[:, s_idx, 0, :, 2, 3] = 0.5  # s far from baseline at this voxel -> not quiescent
    z_hat[:, x_idx, 0, :, 2, 3] = torch.randn(T) * 10  # x noisy here too, but must be exempt

    loss = host._quiescence_consistency_loss(z_hat, tau_s=0.05, tau_f=0.05, eps_x=0.05)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_quiescence_consistency_loss_respects_eps_x_tolerance():
    host = _mk_host()
    B, L, T, H, W = 1, 1, 20, 4, 4
    f_idx, x_idx = host._signal_index("f"), host._signal_index("x")
    z_hat = torch.zeros(B, 7, L, T, H, W)
    z_hat[:, f_idx] = 1.0
    z_hat[:, x_idx, 0, :, 0, 0] = 0.02  # quiescent voxel, |x| below eps_x=0.05

    loss = host._quiescence_consistency_loss(z_hat, tau_s=0.05, tau_f=0.05, eps_x=0.05)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_quiescence_consistency_loss_is_time_resolved_not_whole_trajectory():
    # A voxel quiescent only during PART of the trajectory (correctly flat x there) and
    # genuinely active later (s/f move off baseline, e.g. reached by real diffusion) must
    # not be penalized for that later activity -- the gate is evaluated per timestep, not
    # as a single full-T aggregate, so late-arriving genuine activity is naturally exempt
    # the moment s/f leave baseline (no need to know anything about pulse timing).
    host = _mk_host()
    B, L, T, H, W = 1, 1, 20, 4, 4
    s_idx, f_idx, x_idx = host._signal_index("s"), host._signal_index("f"), host._signal_index("x")
    z_hat = torch.zeros(B, 7, L, T, H, W)
    z_hat[:, f_idx] = 1.0
    half = T // 2
    # First half: quiescent (s=0, f=1) and x correctly flat -- no violation.
    # Second half: genuinely active (s, f move off baseline) and x genuinely nonzero.
    z_hat[:, s_idx, 0, half:, 1, 1] = 0.5
    z_hat[:, f_idx, 0, half:, 1, 1] = 1.4
    z_hat[:, x_idx, 0, half:, 1, 1] = torch.linspace(0, 1, T - half)

    loss = host._quiescence_consistency_loss(z_hat, tau_s=0.05, tau_f=0.05, eps_x=0.05)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_quiescence_consistency_loss_handles_shared_position_scenario_correctly():
    # Two REAL sources sharing the same (h, w) column but living in different layers --
    # the config/simulation/three_layer_single_source_shared_position*.yaml scenario,
    # which is exactly what motivated removing the old cross-layer negative term (it
    # assumed "other layer, same column" always meant "no source," which is false here).
    # This term needs no special-casing to get this right: each source's own s/f are
    # away from baseline (non-quiescent), so neither is checked by the gate at all,
    # regardless of what the other layer at that same column is doing.
    host = _mk_host()
    B, L, T, H, W = 1, 2, 20, 4, 4
    s_idx, f_idx, x_idx = host._signal_index("s"), host._signal_index("f"), host._signal_index("x")
    z_hat = torch.zeros(B, 7, L, T, H, W)
    z_hat[:, f_idx] = 1.0
    # layer-0 source: real dynamics, s/f away from baseline (non-quiescent)
    z_hat[:, s_idx, 0, :, 2, 3] = 0.5
    z_hat[:, x_idx, 0, :, 2, 3] = torch.linspace(0, 1, T)
    # layer-1 source, SAME (h,w): also real dynamics, also non-quiescent
    z_hat[:, s_idx, 1, :, 2, 3] = 0.5
    z_hat[:, x_idx, 1, :, 2, 3] = torch.linspace(1, 0, T)

    loss = host._quiescence_consistency_loss(z_hat, tau_s=0.05, tau_f=0.05, eps_x=0.05)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


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


def test_physics_loss_with_source_layer_uses_per_layer_collocation():
    """Passing source_layer takes the per-layer collocation path and still produces a
    finite loss with the expected per-eq keys."""
    host = _mk_host()
    B, L, T, H, W = 2, 2, 10, 20, 20
    z_hat, dz_hat_dt = _mk_zhat_dzdt(B, 7, L, T, H, W)
    source_position = torch.tensor([[[2, 2], [15, 15]]] * B, dtype=torch.long)  # [B, S=2, 2]
    source_layer = torch.tensor([[0, 1]] * B)
    num_sources = torch.full((B,), 2, dtype=torch.long)
    loss, per_eq = host._physics_loss(
        z_hat,
        dz_hat_dt,
        order="linear",
        lambda_smooth=0.0,
        source_position=source_position,
        source_layer=source_layer,
        num_sources=num_sources,
    )
    assert torch.isfinite(loss)
    assert set(per_eq.keys()) == {"s", "f", "v", "q", "vstar", "qstar"}


def test_physics_loss_full_grid_collocation_routes_flag_to_sampler():
    """loss_config.full_grid_collocation must reach the collocation sampler as
    full_grid=True, not be silently dropped."""
    host = _mk_host(full_grid_collocation=True)
    B, L, T, H, W = 2, 1, 4, 3, 3
    z_hat, dz_hat_dt = _mk_zhat_dzdt(B, 5, L, T, H, W)
    source_position = torch.randint(0, 3, (B, 1, 2))
    num_sources = torch.ones(B, dtype=torch.long)

    calls = []
    real_fn = MICHLossMixin._sample_collocation_indices

    def spy(**kwargs):
        calls.append(kwargs.get("full_grid"))
        return real_fn(**kwargs)

    host._sample_collocation_indices = spy
    try:
        host._physics_loss(
            z_hat,
            dz_hat_dt,
            order="linear",
            lambda_smooth=0.0,
            source_position=source_position,
            num_sources=num_sources,
        )
    finally:
        del host._sample_collocation_indices

    assert calls == [True]


def test_data_loss_full_grid_collocation_routes_flag_to_sampler():
    host = _mk_host(full_grid_collocation=True)
    host.normaliser = None
    B, L, T, H, W = 2, 1, 4, 3, 3
    z_hat, _ = _mk_zhat_dzdt(B, 5, L, T, H, W)
    bold_norm = torch.randn(B, L, T, H, W)
    source_position = torch.randint(0, 3, (B, 1, 2))
    num_sources = torch.ones(B, dtype=torch.long)

    calls = []
    real_fn = MICHLossMixin._sample_collocation_indices

    def spy(**kwargs):
        calls.append(kwargs.get("full_grid"))
        return real_fn(**kwargs)

    host._sample_collocation_indices = spy
    try:
        host._data_loss(z_hat, bold_norm, source_position=source_position, num_sources=num_sources)
    finally:
        del host._sample_collocation_indices

    assert calls == [True]


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
# _supervision_loss / _derivative_supervision_loss / _x_phase_loss
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
    source_layer = torch.zeros(B, 1, dtype=torch.long)
    num_sources = torch.ones(B, dtype=torch.long)
    total, per_sig = host._supervision_loss(
        z_hat, batch, source_position, source_layer, num_sources
    )
    assert torch.isclose(total, torch.tensor(0.0), atol=1e-6)
    assert set(per_sig.keys()) == {"s", "f", "v", "q", "vstar", "qstar"}


def test_supervision_loss_nonzero_for_mismatched_signals():
    host = _mk_host()
    B, L, T, H, W = 2, 1, 10, 5, 5
    batch = _mk_supervision_batch(B, L, T, H, W)
    z_hat = torch.randn(B, 7, L, T, H, W)  # unrelated to batch
    source_position = torch.randint(0, 5, (B, 1, 2))
    source_layer = torch.zeros(B, 1, dtype=torch.long)
    num_sources = torch.ones(B, dtype=torch.long)
    total, _ = host._supervision_loss(z_hat, batch, source_position, source_layer, num_sources)
    assert total > 0.0


def test_supervision_loss_handles_t_min_mismatch():
    host = _mk_host()
    B, L, H, W = 2, 1, 5, 5
    z_hat = torch.randn(B, 7, L, 12, H, W)  # T=12
    batch = _mk_supervision_batch(B, L, 8, H, W)  # T_latent=8 < z_hat's T
    source_position = torch.randint(0, 5, (B, 1, 2))
    source_layer = torch.zeros(B, 1, dtype=torch.long)
    num_sources = torch.ones(B, dtype=torch.long)
    total, _ = host._supervision_loss(z_hat, batch, source_position, source_layer, num_sources)
    assert torch.isfinite(total)


def test_supervision_loss_perfect_match_gives_zero_multilayer_distinct_sources():
    """Per-layer collocation + dense-source components, both zero on a perfect match,
    across multiple layers each with their own distinct source."""
    host = _mk_host(supervision_loss=None)
    B, L, T, H, W = 2, 2, 10, 20, 20
    batch = _mk_supervision_batch(B, L, T, H, W)
    z_hat = torch.zeros(B, 7, L, T, H, W)
    for sig, bk in MICHLossMixin._SUPERVISION_KEYS_FULL:
        z_hat[:, host._signal_index(sig)] = batch[bk]
    source_position = torch.tensor([[[2, 2], [15, 15]]] * B, dtype=torch.long)  # [B, S=2, 2]
    source_layer = torch.tensor([[0, 1]] * B)  # source 0 in layer 0, source 1 in layer 1
    num_sources = torch.full((B,), 2, dtype=torch.long)
    total, per_sig = host._supervision_loss(
        z_hat, batch, source_position, source_layer, num_sources
    )
    assert torch.isclose(total, torch.tensor(0.0), atol=1e-6)
    assert set(per_sig.keys()) == {"s", "f", "v", "q", "vstar", "qstar"}


def test_supervision_loss_dense_component_masks_padded_sources_and_batch_entries():
    """The dense (source-voxel) component ignores padded source slots and samples with
    zero valid sources, same masking guarantee the old _source_supervision_loss had."""
    host = _mk_host(supervision_loss=None)
    B, S, L, T, H, W = 2, 2, 1, 10, 5, 5
    batch = _mk_supervision_batch(B, L, T, H, W)
    z_hat = torch.zeros(B, 7, L, T, H, W)
    for sig, bk in MICHLossMixin._SUPERVISION_KEYS_FULL:
        z_hat[:, host._signal_index(sig)] = batch[bk]
    source_position = torch.randint(0, 5, (B, S, 2))
    source_layer = torch.zeros(B, S, dtype=torch.long)
    num_sources = torch.tensor([1, 0])  # sample 1 has zero valid sources
    total, _ = host._supervision_loss(z_hat, batch, source_position, source_layer, num_sources)
    assert torch.isclose(total, torch.tensor(0.0), atol=1e-6)


def test_supervision_loss_dense_component_is_layer_scoped():
    """A source's dense component only compares its own layer -- a mismatch confined to
    a different layer at the same (h, w) must not affect the loss. The grid/collocation
    component is mocked to a fixed point well away from the corruption, since its own
    random draws could otherwise legitimately land on the same (h, w) in that other
    layer and pick up the same mismatch for an unrelated (correct) reason."""
    host = _mk_host(supervision_loss=None)
    B, L, T, H, W = 1, 2, 10, 10, 10
    batch = _mk_supervision_batch(B, L, T, H, W)
    z_hat = torch.zeros(B, 7, L, T, H, W)
    for sig, bk in MICHLossMixin._SUPERVISION_KEYS_FULL:
        z_hat[:, host._signal_index(sig)] = batch[bk]
    # Corrupt layer 1's prediction at the same (h, w) as layer 0's source -- should be
    # irrelevant to a source whose source_layer says it belongs to layer 0.
    s_idx = host._signal_index("s")
    z_hat[:, s_idx, 1, :, 2, 2] += 100.0
    source_position = torch.tensor([[[2, 2]]], dtype=torch.long)
    source_layer = torch.tensor([[0]])
    num_sources = torch.ones(B, dtype=torch.long)

    from mich.models.collocation import CollocationBatch

    fixed_idx = CollocationBatch(
        t=torch.zeros(1, 1, 1, 1, dtype=torch.long),
        h=torch.full((B, 1, 1, 1), 7, dtype=torch.long),  # away from the corrupted (2, 2)
        w=torch.full((B, 1, 1, 1), 7, dtype=torch.long),
    )
    with patch.object(
        MICHLossMixin,
        "_sample_collocation_indices_per_layer",
        return_value=[fixed_idx, fixed_idx],
    ):
        _total, per_sig = host._supervision_loss(
            z_hat, batch, source_position, source_layer, num_sources
        )
    assert torch.isclose(per_sig["s"], torch.tensor(0.0), atol=1e-6)


def test_supervision_loss_grid_uses_grid_fn_and_dense_uses_dense_fn():
    """The grid component's gather is [B, n_times, n_space] with independently
    scattered (t, h, w) points along dim=1 -- not one location's trajectory -- so it
    must go through _supervision_grid_loss_fn (Pearson-free), not
    _supervision_loss_fn (used correctly for the dense component, which really does
    gather one fixed location's full T_min)."""
    host = _mk_host()
    B, L, T, H, W = 2, 1, 10, 5, 5
    batch = _mk_supervision_batch(B, L, T, H, W)
    z_hat = torch.randn(B, 7, L, T, H, W)
    source_position = torch.randint(0, 5, (B, 1, 2))
    source_layer = torch.zeros(B, 1, dtype=torch.long)
    num_sources = torch.ones(B, dtype=torch.long)

    grid_calls, dense_calls = [], []
    real_grid_fn, real_dense_fn = host._supervision_grid_loss_fn, host._supervision_loss_fn

    def spy_grid(pred, true):
        grid_calls.append(pred.shape)
        return real_grid_fn(pred, true)

    def spy_dense(pred, true):
        dense_calls.append(pred.shape)
        return real_dense_fn(pred, true)

    host._supervision_grid_loss_fn, host._supervision_loss_fn = spy_grid, spy_dense
    try:
        host._supervision_loss(z_hat, batch, source_position, source_layer, num_sources)
    finally:
        host._supervision_grid_loss_fn = real_grid_fn
        host._supervision_loss_fn = real_dense_fn

    n_sig = len(MICHLossMixin._SUPERVISION_KEYS_FULL)
    assert len(grid_calls) == n_sig * L  # one collocation call per layer, per signal
    assert all(len(shape) == 3 for shape in grid_calls)  # [B, n_times, n_space]
    assert len(dense_calls) == n_sig  # one dense call per signal
    assert all(len(shape) == 2 for shape in dense_calls)  # [M, T_min]


def test_supervision_loss_full_grid_collocation_routes_flag_to_sampler():
    """loss_config.full_grid_collocation must reach the (always per-layer)
    collocation sampler _supervision_loss uses, as full_grid=True."""
    host = _mk_host(full_grid_collocation=True)
    B, L, T, H, W = 2, 1, 6, 3, 3
    batch = _mk_supervision_batch(B, L, T, H, W)
    z_hat = torch.randn(B, 7, L, T, H, W)
    source_position = torch.randint(0, 3, (B, 1, 2))
    source_layer = torch.zeros(B, 1, dtype=torch.long)
    num_sources = torch.ones(B, dtype=torch.long)

    calls = []
    real_fn = MICHLossMixin._sample_collocation_indices_per_layer

    def spy(**kwargs):
        calls.append(kwargs.get("full_grid"))
        return real_fn(**kwargs)

    host._sample_collocation_indices_per_layer = spy
    try:
        host._supervision_loss(z_hat, batch, source_position, source_layer, num_sources)
    finally:
        del host._sample_collocation_indices_per_layer

    assert calls == [True]


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


def test_derivative_supervision_loss_is_layer_scoped():
    """When source_layer is given, a source's term only compares its own layer -- a
    mismatch confined to a different layer at the same (h, w) must not affect it."""
    host = _mk_host(order="linear")
    B, L, T, H, W = 1, 2, 10, 5, 5
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

    # Corrupt layer 1's dz_hat_dt at the same (h, w) as the (layer-0) source -- should
    # be irrelevant to a source whose source_layer says it belongs to layer 0.
    s_idx = host._signal_index("s")
    dz_hat_dt[:, s_idx, 1, :, 2, 2] += 100.0

    source_position = torch.tensor([[[2, 2]]])
    source_layer = torch.tensor([[0]])
    num_sources = torch.ones(B, dtype=torch.long)
    total, _ = host._derivative_supervision_loss(
        dz_hat_dt, batch, source_position, num_sources, source_layer
    )
    assert torch.isclose(total, torch.tensor(0.0), atol=1e-4)


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


def test_derivative_supervision_loss_drain_coupling_matches_analytic_target():
    """Multi-layer (has_drain) case: layer>0's v/q analytic target must pick up the
    same lambda_d * vstar/qstar-from-the-layer-below term _compute_physics_layer_loss
    adds, and vstar/qstar get their own ground-truth-only analytic targets."""
    host = _mk_host(order="linear", dzdt_supervision_signals=("v", "q", "vstar", "qstar"))
    B, L, T, H, W = 2, 2, 10, 5, 5
    # Read the constants off the host rather than restating them, so this oracle
    # keeps testing the *structure* of the ODE targets even when the fixture
    # values change. Hardcoding them here previously made the test pass only
    # because tau and tau_d happened to be 1.0, which hid every `/ tau` bug.
    lambda_d = host.hparams.haemo.lambda_d
    tau_d = host.hparams.haemo.tau_d
    tau = host._physio("tau")

    f_true = torch.rand(B, L, T, H, W) + 0.5
    v_true = torch.rand(B, L, T, H, W) + 0.5
    q_true = torch.rand(B, L, T, H, W) + 0.5
    v_star_true = torch.rand(B, L, T, H, W) + 0.5
    q_star_true = torch.rand(B, L, T, H, W) + 0.5
    batch = {
        "s": torch.randn(B, L, T, H, W),
        "f": f_true,
        "v": v_true,
        "q": q_true,
        "v_star": v_star_true,
        "q_star": q_star_true,
    }

    vdot, qdot = host._balloon_v_q_dot_targets(f_true, v_true, q_true, "linear")
    drain_v = torch.zeros_like(vdot)
    drain_v[:, 1:] = lambda_d * v_star_true[:, :-1]
    drain_q = torch.zeros_like(qdot)
    drain_q[:, 1:] = lambda_d * q_star_true[:, :-1]
    analytic = {
        "v": (vdot + drain_v) / tau,
        "q": (qdot + drain_q) / tau,
        "vstar": (-v_star_true + v_true - 1) / tau_d,
        "qstar": (-q_star_true + q_true - 1) / tau_d,
    }
    dz_hat_dt = torch.zeros(B, 7, L, T, H, W)
    t_norm_to_physical = T - 1
    for sig in ("v", "q", "vstar", "qstar"):
        dz_hat_dt[:, host._signal_index(sig)] = analytic[sig] * t_norm_to_physical

    source_position = torch.randint(0, 5, (B, 1, 2))
    num_sources = torch.ones(B, dtype=torch.long)
    total, per_sig = host._derivative_supervision_loss(
        dz_hat_dt, batch, source_position, num_sources
    )
    assert torch.isclose(total, torch.tensor(0.0), atol=1e-4)
    assert set(per_sig.keys()) == {"v", "q", "vstar", "qstar"}


def test_derivative_supervision_loss_vstar_qstar_without_drain_raises():
    host = _mk_host(order="linear", dzdt_supervision_signals=("vstar",))
    B, L, T, H, W = 2, 1, 10, 5, 5
    batch = {
        "s": torch.randn(B, L, T, H, W),
        "f": torch.rand(B, L, T, H, W) + 0.5,
        "v": torch.rand(B, L, T, H, W) + 0.5,
        "q": torch.rand(B, L, T, H, W) + 0.5,
    }
    dz_hat_dt = torch.randn(B, 5, L, T, H, W)  # no drain channels
    source_position = torch.randint(0, 5, (B, 1, 2))
    num_sources = torch.ones(B, dtype=torch.long)
    with pytest.raises(ValueError, match="vstar/qstar"):
        host._derivative_supervision_loss(dz_hat_dt, batch, source_position, num_sources)


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


# -----------------------------
# Phase 2 oracles: ODE-residual, BOLD value, and loss-composition contracts
#
# The tests above this block establish that the physics losses run, return the
# right keys, and stay finite. They do not pin a single numeric value, so a
# swapped coefficient or a dropped `/ tau` passes them all. The tests below
# supply the missing oracles: a state that exactly satisfies the balloon ODE
# system must drive every residual to zero, a hand-computed BOLD value must be
# reproduced exactly, and the documented `total = colloc + lambda_src * src`
# contract must actually hold.
# -----------------------------


def _consistent_physics_state(host, B, L, T, H, W, *, order, n_channels=7, seed=0):
    """Build a (z_hat, dz_hat_dt) pair that exactly satisfies the balloon ODE system.

    Every residual `_compute_physics_layer_loss` forms is
    `analytic_derivative - ODE_right_hand_side`, so feeding it a state whose
    derivatives *are* the right-hand sides must yield exactly zero for every
    equation. That makes this an independent oracle: it constrains each
    coefficient, each `/ tau`, and each drain coupling term individually,
    without restating the implementation's own expression for them.

    States are kept in a benign range (f/v/q near 1, x/s small) so that
    `_sanitise_states`' NaN replacement and 0.1 floor are no-ops and cannot mask
    a mismatch.
    """
    gen = torch.Generator().manual_seed(seed)

    def _rand(scale, offset=0.0):
        return torch.rand(B, L, T, H, W, generator=gen) * scale + offset

    x = _rand(0.6, -0.3)
    s = _rand(0.6, -0.3)
    f = _rand(0.6, 0.7)  # [0.7, 1.3]
    v = _rand(0.6, 0.7)
    q = _rand(0.6, 0.7)
    v_star = _rand(0.4, -0.2)
    q_star = _rand(0.4, -0.2)

    has_drain = n_channels > 5
    kappa = host._physio("kappa")
    gamma = host._physio("gamma")
    tau = host._physio("tau")
    tau_d = host.hparams.haemo.tau_d
    lambda_d = host.hparams.haemo.lambda_d

    vdot, qdot = host._balloon_v_q_dot_targets(f, v, q, order)

    # The flow-inducing signal and flow equations.
    ds_dt = x - kappa * s - gamma * (f - 1)
    df_dt = s

    # Volume/deoxyhaemoglobin, including the inter-layer drainage a layer picks
    # up from the layer below it (layer 0 has nothing below it, so no term).
    dv_dt = vdot.clone()
    dq_dt = qdot.clone()
    if has_drain and L > 1:
        dv_dt[:, 1:] = dv_dt[:, 1:] + lambda_d * v_star[:, :-1]
        dq_dt[:, 1:] = dq_dt[:, 1:] + lambda_d * q_star[:, :-1]
    dv_dt = dv_dt / tau
    dq_dt = dq_dt / tau

    dv_star_dt = (-v_star + v - 1) / tau_d
    dq_star_dt = (-q_star + q - 1) / tau_d

    channels = [x, s, f, v, q, v_star, q_star][:n_channels]
    z_hat = torch.stack(channels, dim=1)

    # `_compute_physics_layer_loss` converts the derivative side from the
    # decoder's [0, 1]-normalised time grid to per-index units by dividing by
    # (T - 1), so pre-multiply here to cancel that exactly.
    t_norm_to_physical = T - 1
    zero = torch.zeros_like(x)
    grads = [zero, ds_dt, df_dt, dv_dt, dq_dt, dv_star_dt, dq_star_dt][:n_channels]
    dz_hat_dt = torch.stack(grads, dim=1) * t_norm_to_physical

    return z_hat, dz_hat_dt


def _full_grid_idx(B, T, H, W):
    """A CollocationBatch covering every (t, h, w) once, so the oracle below is
    checked at every point rather than a random handful."""
    from mich.models.collocation import CollocationBatch

    t = torch.arange(T).view(1, 1, T, 1).expand(B, 1, T, H * W)
    hw = torch.arange(H * W)
    h = (hw // W).view(1, 1, 1, H * W).expand(B, 1, T, H * W)
    w = (hw % W).view(1, 1, 1, H * W).expand(B, 1, T, H * W)
    return CollocationBatch(t=t.contiguous(), h=h.contiguous(), w=w.contiguous())


@pytest.mark.parametrize("order", ["exact", "linear", "quadratic"])
def test_compute_physics_layer_loss_is_zero_for_ode_consistent_state_no_drain(order):
    """An ODE-consistent state must drive every per-equation residual to zero.

    This is the oracle the key-set-and-finiteness tests above are missing: it
    pins the s/f/v/q right-hand sides for all three balloon orders, including
    the quadratic branch, whose coefficients all cancel at the resting state and
    were therefore never constrained by any previous test.
    """
    host = _mk_host(order=order)
    B, L, T, H, W = 2, 1, 6, 3, 4
    z_hat, dz_hat_dt = _consistent_physics_state(host, B, L, T, H, W, order=order, n_channels=5)
    idx = _full_grid_idx(B, T, H, W)

    losses = host._compute_physics_layer_loss(
        z_hat, dz_hat_dt, idx, layer=0, burn_in=0, order=order
    )

    assert set(losses.keys()) == {"s", "f", "v", "q"}
    for eq, value in losses.items():
        assert value.abs() < 1e-10, f"{order}/{eq} residual should vanish, got {value.item():.3e}"


@pytest.mark.parametrize("order", ["exact", "linear", "quadratic"])
@pytest.mark.parametrize("layer", [0, 1])
def test_compute_physics_layer_loss_is_zero_for_ode_consistent_state_with_drain(order, layer):
    """Same oracle in multi-layer (drain) mode, per layer.

    Layer 1 must pick up the `lambda_d * vstar/qstar` term from layer 0 while
    layer 0 must not, and vstar/qstar get their own `(-v_star + v - 1) / tau_d`
    targets. That `tau_d` divisor is written twice in the source (here and in
    `_derivative_supervision_loss`); only the other copy was pinned before.
    """
    host = _mk_host(order=order)
    B, L, T, H, W = 2, 2, 6, 3, 4
    z_hat, dz_hat_dt = _consistent_physics_state(host, B, L, T, H, W, order=order, n_channels=7)
    idx = _full_grid_idx(B, T, H, W)

    losses = host._compute_physics_layer_loss(
        z_hat, dz_hat_dt, idx, layer=layer, burn_in=0, order=order
    )

    assert set(losses.keys()) == {"s", "f", "v", "q", "vstar", "qstar"}
    for eq, value in losses.items():
        assert value.abs() < 1e-10, (
            f"{order}/layer{layer}/{eq} residual should vanish, got {value.item():.3e}"
        )


def test_compute_physics_layer_loss_v_q_targets_scale_with_tau():
    """The v/q targets must be divided by tau, not multiplied by it or left raw.

    Built as a differential check against the zero-residual oracle: a state made
    consistent for one tau must be inconsistent for a different tau, and the
    residual must grow in the direction `/ tau` predicts.
    """
    host = _mk_host(order="linear")
    B, L, T, H, W = 2, 1, 6, 3, 4
    tau = host._physio("tau")
    assert tau != 1.0, "fixture must use a non-unit tau or this test proves nothing"

    z_hat, dz_hat_dt = _consistent_physics_state(host, B, L, T, H, W, order="linear", n_channels=5)
    idx = _full_grid_idx(B, T, H, W)

    baseline = host._compute_physics_layer_loss(
        z_hat, dz_hat_dt, idx, layer=0, burn_in=0, order="linear"
    )
    assert baseline["v"].abs() < 1e-10

    # Scale only the v/q derivative channels by tau. The targets are `raw / tau`,
    # so a derivative of `raw` (i.e. tau times too large) must no longer match.
    v_idx, q_idx = host._signal_index("v"), host._signal_index("q")
    perturbed = dz_hat_dt.clone()
    perturbed[:, v_idx] *= tau
    perturbed[:, q_idx] *= tau

    off = host._compute_physics_layer_loss(
        z_hat, perturbed, idx, layer=0, burn_in=0, order="linear"
    )
    assert off["v"] > 1e-6, "v residual must react to a tau-sized error in dv/dt"
    assert off["q"] > 1e-6, "q residual must react to a tau-sized error in dq/dt"
    # s/f do not involve tau and must be untouched.
    assert off["s"].abs() < 1e-10
    assert off["f"].abs() < 1e-10


def test_compute_bold_matches_hand_computed_value():
    """Pin `_compute_bold` against arithmetic done by hand, with k1/k2/k3 all
    distinct so that permuting the three terms changes the result.

    The existing formula test re-derives the expression in the test body, which
    cannot distinguish a wrong formula from a faithfully-copied wrong formula.
    Here the inputs are chosen so every term is a clean number:
        k1 * (1 - q)     = 1.0 * (1 - 0.5)         =  0.5
        k2 * (1 - q / v) = 2.0 * (1 - 0.5 / 0.25)  = -2.0
        k3 * (1 - v)     = 4.0 * (1 - 0.25)        =  3.0
        V0 * sum         = 0.02 * 1.5              =  0.03
    """
    acquisition = _mk_acquisition(k1=1.0, k2=2.0, k3=4.0)
    q = torch.tensor([0.5])
    v = torch.tensor([0.25])

    out = MICHLossMixin._compute_bold(q=q, v=v, acquisition=acquisition, V0=0.02)

    assert torch.isclose(out, torch.tensor(0.03), atol=1e-7), out


def test_compute_bold_increases_as_deoxyhaemoglobin_falls():
    """Physiological sign convention: with volume held fixed, BOLD signal must
    rise as deoxyhaemoglobin q falls (washout increases signal)."""
    acquisition = _mk_acquisition()
    v = torch.ones(1)
    high_q = MICHLossMixin._compute_bold(
        q=torch.tensor([1.2]), v=v, acquisition=acquisition, V0=0.02
    )
    low_q = MICHLossMixin._compute_bold(
        q=torch.tensor([0.8]), v=v, acquisition=acquisition, V0=0.02
    )
    assert low_q > high_q


def test_data_loss_total_equals_colloc_plus_lambda_src_times_src():
    """The documented contract `total = colloc_loss + lambda_src * src_loss`.

    Every previous `_data_loss` test asserted only that the total was a finite
    scalar, which a sign flip or a dropped weight satisfies just as well. This
    needs a non-unit lambda_src in the fixture to have any teeth.
    """
    host = _mk_host()
    host.normaliser = None
    lambda_src = host.hparams.loss_config.lambda_src
    assert lambda_src != 1.0, "fixture must use a non-unit lambda_src"

    B, L, T, H, W = 2, 2, 10, 5, 6
    torch.manual_seed(0)
    z_hat = torch.randn(B, 7, L, T, H, W) * 0.1 + 1.0
    bold_norm = torch.randn(B, L, T, H, W)
    source_position = torch.tensor([[[1, 3]], [[4, 2]]], dtype=torch.long)
    source_layer = torch.tensor([[0], [1]])
    num_sources = torch.ones(B, dtype=torch.long)

    total, colloc_loss, src_loss = host._data_loss(
        z_hat,
        bold_norm,
        source_position=source_position,
        num_sources=num_sources,
        source_layer=source_layer,
    )

    assert torch.isclose(total, colloc_loss + lambda_src * src_loss, atol=1e-6)
    # Guard against the contract holding only because src_loss vanished.
    assert src_loss > 0
