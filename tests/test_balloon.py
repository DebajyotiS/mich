from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
import torch
from mich.data.balloon import (  # noqa: F401
    AcquisitionConstants,
    BoldPostProcessingConfig,
    CortexLayer,
    HaemodynamicConstants,
    HaemodynamicState,
    NoiseModel,
    PointSpreadFunction,
    _apply_psf_numpy,
    _apply_psf_torch,
    _clamp_any,
    _clamp_pos,
    _finite,
    _gaussian_kernel_1d,
    _generate_bold_noise,
    _get_torch_ref,
    _is_torch,
    _reflect_pad,
    balloon_derivatives,
    delay_filter_derivatives,
    get_bold_from_state,
    get_inversion_derivatives,
    rk4_step,
    sanitize_state,
    simulate_cortex,
)

# -----------------------------
# Test doubles / helpers
# -----------------------------


@dataclass(frozen=True)
class FakeNoise:
    type: str
    seed: int | None = None


def _mk_consts() -> HaemodynamicConstants:
    return HaemodynamicConstants(
        kappa=0.65,
        gamma=0.41,
        alpha=0.32,
        E0=0.34,
        V0=0.02,
    )


def _mk_acq() -> AcquisitionConstants:
    return AcquisitionConstants(k1=7.0, k2=2.0, k3=2.0)


def _mk_layer(
    depth=0,
    tau=1.0,
    *,
    x=0.0,
    s=0.0,
    f=1.0,
    v=1.0,
    q=1.0,
    v_star: Any = None,
    q_star: Any = None,
    lambda_d: float = 0.0,
    drain_from: CortexLayer | None = None,
) -> CortexLayer:
    st = HaemodynamicState(
        x=np.array(x, dtype=np.float64),
        s=np.array(s, dtype=np.float64),
        f=np.array(f, dtype=np.float64),
        v=np.array(v, dtype=np.float64),
        q=np.array(q, dtype=np.float64),
        v_star=v_star,
        q_star=q_star,
    )
    return CortexLayer(depth=depth, tau=tau, state=st, lambda_d=lambda_d, drain_from=drain_from)


def _np_state_scalar(**kwargs) -> dict[str, np.ndarray]:
    # convenience dict of scalar float64 arrays
    out = {}
    for k, v in kwargs.items():
        out[k] = np.array(v, dtype=np.float64)
    return out


# -----------------------------
# PointSpreadFunction + kernels
# -----------------------------


def test_gaussian_kernel_1d_normalized_and_symmetric():
    k = _gaussian_kernel_1d(sigma=2.0, truncate=4.0)
    assert k.ndim == 1
    assert np.isclose(k.sum(), 1.0, atol=1e-12)
    assert np.allclose(k, k[::-1])


def test_psf_sigma_matches_fwhm_formula():
    psf = PointSpreadFunction(fwhm=3.0)
    expected = 3.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    assert np.isclose(psf.sigma, expected)


def test_psf_apply_noop_when_fwhm_nonpositive_numpy_and_torch():
    psf0 = PointSpreadFunction(fwhm=0.0)
    x_np = np.random.default_rng(0).standard_normal((5, 8, 8)).astype(np.float64)
    x_th = torch.as_tensor(x_np)

    y_np = psf0.apply(x_np)
    y_th = psf0.apply(x_th)

    assert y_np is x_np  # implementation returns input directly
    assert y_th is x_th


def test_apply_psf_numpy_blurs_only_spatial_axes_not_time():
    # Impulse at t=2, center pixel; after blur, time axis should remain impulse-only at t=2
    T, H, W = 5, 21, 21
    x = np.zeros((T, H, W), dtype=np.float64)
    x[2, H // 2, W // 2] = 1.0

    y = _apply_psf_numpy(x, sigma=2.0)
    assert y.shape == x.shape

    # still all zeros at other times
    assert np.allclose(y[0], 0.0)
    assert np.allclose(y[1], 0.0)
    assert np.allclose(y[3], 0.0)
    assert np.allclose(y[4], 0.0)

    # at t=2, spatial spread occurs: center reduced, neighbors > 0
    assert 0.0 < y[2].max() < 1.0
    assert y[2, H // 2, W // 2] == y[2].max()
    assert np.sum(y[2] > 0) > 1


def test_apply_psf_torch_matches_numpy_approximately_for_2d_interior():
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((7, 25, 25)).astype(np.float64)
    sigma = 1.5

    y_np = _apply_psf_numpy(x_np, sigma=sigma)
    y_th = _apply_psf_torch(torch.as_tensor(x_np), sigma=sigma).cpu().numpy()

    # ignore boundaries because scipy gaussian_filter uses reflect mode by default,
    # while torch conv2d uses zero-padding.
    pad = len(_gaussian_kernel_1d(sigma)) // 2
    if pad == 0:
        assert np.allclose(y_th, y_np, atol=1e-2, rtol=1e-2)
        return

    sl = (slice(None), slice(pad, -pad), slice(pad, -pad))
    assert np.allclose(y_th[sl], y_np[sl], atol=1e-2, rtol=1e-2)


def test_apply_psf_torch_1d_spatial_supported_and_3d_spatial_not_implemented():
    # 1D spatial (T, X)
    x = torch.zeros((5, 31), dtype=torch.float64)
    x[2, 15] = 1.0
    y = _apply_psf_torch(x, sigma=2.0)
    assert y.shape == x.shape
    assert y[2].max() < 1.0  # blurred

    # 3D spatial (T, X, Y, Z) => n_spatial=3 should raise
    x3 = torch.zeros((5, 9, 9, 9), dtype=torch.float64)
    with pytest.raises(NotImplementedError, match="PSF not implemented"):
        _apply_psf_torch(x3, sigma=1.0)


def test_apply_psf_numpy_reflect_boundary_preserves_more_energy_than_absorbing():
    # Impulse one pixel from the top edge (not exactly at the edge, which is a
    # degenerate fixed point of the "reflect" convention -- see torch test below).
    T, H, W = 3, 11, 11
    x = np.zeros((T, H, W), dtype=np.float64)
    x[1, 1, 5] = 1.0

    y_reflect = _apply_psf_numpy(x, sigma=1.5, boundary="reflect")
    y_absorb = _apply_psf_numpy(x, sigma=1.5, boundary="absorbing")

    assert y_reflect.shape == x.shape
    assert not np.allclose(y_reflect[1], y_absorb[1])
    # reflect boundary must not zero-out mass the way zero-padding does
    assert y_reflect[1].sum() > y_absorb[1].sum()
    # scipy's mode="reflect" duplicates the edge sample, which exactly conserves mass
    assert np.isclose(y_reflect[1].sum(), 1.0, atol=1e-8)


def test_apply_psf_torch_reflect_boundary_preserves_more_energy_than_absorbing_1d():
    T, X = 5, 11
    x = torch.zeros((T, X), dtype=torch.float64)
    x[2, 1] = 1.0

    y_reflect = _apply_psf_torch(x, sigma=1.5, boundary="reflect")
    y_absorb = _apply_psf_torch(x, sigma=1.5, boundary="absorbing")

    assert y_reflect.shape == x.shape
    assert not torch.allclose(y_reflect[2], y_absorb[2])
    assert y_reflect[2].sum().item() > y_absorb[2].sum().item()


def test_apply_psf_torch_reflect_boundary_preserves_more_energy_than_absorbing_2d():
    T, H, W = 3, 11, 11
    x = torch.zeros((T, H, W), dtype=torch.float64)
    x[1, 1, 5] = 1.0

    y_reflect = _apply_psf_torch(x, sigma=1.5, boundary="reflect")
    y_absorb = _apply_psf_torch(x, sigma=1.5, boundary="absorbing")

    assert y_reflect.shape == x.shape
    assert not torch.allclose(y_reflect[1], y_absorb[1])
    assert y_reflect[1].sum().item() > y_absorb[1].sum().item()


def test_apply_psf_torch_reflect_boundary_degenerates_to_absorbing_at_true_edge_pixel():
    # NOTE: this pins down a real quirk of the current implementation rather than
    # asserting "nice" behaviour: _reflect_pad implements a non-duplicating reflect
    # (matches np.pad(mode="reflect"), where the edge sample is a fixed point of the
    # reflection). For an impulse placed exactly at index 0, none of its reflected
    # copies land back inside the padding window, so boundary="reflect" is bit-for-bit
    # identical to boundary="absorbing" here -- unlike the numpy path (scipy
    # mode="reflect", which duplicates the edge sample and does conserve mass even at
    # index 0). See report for discussion of the resulting numpy/torch inconsistency.
    T, X = 5, 11
    x = torch.zeros((T, X), dtype=torch.float64)
    x[2, 0] = 1.0

    y_reflect = _apply_psf_torch(x, sigma=1.5, boundary="reflect")
    y_absorb = _apply_psf_torch(x, sigma=1.5, boundary="absorbing")
    assert torch.allclose(y_reflect, y_absorb)


def test_reflect_pad_matches_numpy_pad_reflect_1d():
    x = torch.arange(4, dtype=torch.float64).reshape(1, 1, 4)  # [0, 1, 2, 3]
    padded = _reflect_pad(x, pad=2, dim=2).flatten().numpy()
    expected = np.pad(np.arange(4, dtype=np.float64), (2, 2), mode="reflect")
    assert np.array_equal(padded, expected)
    assert np.array_equal(padded, np.array([2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 2.0, 1.0]))


# -----------------------------
# NoiseModel
# -----------------------------


def test_noise_model_preset_valid_and_invalid():
    nm = NoiseModel.preset("3T", V=1.0, TR=2.0, nT=1, differential=False, S0=1.0)
    assert isinstance(nm, NoiseModel)

    with pytest.raises(ValueError, match="Unknown preset"):
        NoiseModel.preset("9T", V=1.0, TR=2.0)


def test_noise_model_sigma_raises_for_multiple_measurements_without_differential():
    nm = NoiseModel.preset("3T", V=1.0, TR=2.0, nT=4, differential=False)
    with pytest.raises(ValueError, match="Multiple measurements"):
        _ = nm.sigma


def test_noise_model_sigma_and_noise_std_finite_and_positive():
    nm = NoiseModel.preset("7T", V=1.2, TR=2.0, nT=1, differential=False, S0=3.0)
    assert nm.sigma > 0.0
    assert np.isfinite(nm.sigma)
    assert np.isclose(nm.noise_std, nm.sigma * nm.S0)


def test_noise_model_sigma_differential_branch_runs():
    nm = NoiseModel.preset("3T", V=1.0, TR=1.0, nT=10, differential=True)
    s = nm.sigma
    assert s > 0.0
    assert np.isfinite(s)


# -----------------------------
# _generate_bold_noise
# -----------------------------


@pytest.mark.parametrize("noise_type", ["white", "uniform", "pink"])
def test_generate_bold_noise_reproducible_with_seed(noise_type):
    noise = FakeNoise(type=noise_type, seed=123)
    a = _generate_bold_noise((32, 4, 3), noise, amplitude=0.7)
    b = _generate_bold_noise((32, 4, 3), noise, amplitude=0.7)
    assert a.shape == (32, 4, 3)
    assert a.dtype == np.float64
    assert np.allclose(a, b)


def test_generate_bold_noise_uniform_range():
    noise = FakeNoise(type="uniform", seed=0)
    amp = 0.3
    x = _generate_bold_noise((17, 5), noise, amplitude=amp)
    assert x.min() >= -amp - 1e-12
    assert x.max() <= amp + 1e-12


def test_generate_bold_noise_pink_steps_1_edge_case():
    noise = FakeNoise(type="pink", seed=0)
    x = _generate_bold_noise((1, 7, 3), noise, amplitude=1.0)
    assert x.shape == (1, 7, 3)
    assert np.isfinite(x).all()


def test_generate_bold_noise_unknown_type_raises():
    noise = FakeNoise(type="nope", seed=0)
    with pytest.raises(ValueError, match="Unknown noise type"):
        _generate_bold_noise((8, 2, 2), noise, amplitude=1.0)


# -----------------------------
# Small utility functions
# -----------------------------


def test_is_torch_and_get_torch_ref():
    xs = [np.zeros((2,)), torch.zeros((2,))]
    assert _is_torch(xs[0]) is False
    assert _is_torch(xs[1]) is True
    ref = _get_torch_ref(xs)
    assert isinstance(ref, torch.Tensor)


def test_finite_clamp_pos_and_clamp_any_numpy_and_torch():
    x_np = np.array([np.nan, np.inf, -np.inf, -2.0, 0.0, 2.0], dtype=np.float64)
    y_np = _finite(x_np, max_abs=10.0)
    assert np.isfinite(y_np).all()
    assert y_np[0] == 0.0
    assert y_np[1] == 10.0
    assert y_np[2] == -10.0

    z_np = _clamp_pos(np.array([-1.0, 0.0, 1.0], dtype=np.float64), eps=1e-3, max_abs=10.0)
    assert np.all(z_np >= 1e-3)

    w_np = _clamp_any(np.array([-20.0, 0.0, 20.0], dtype=np.float64), max_abs=5.0)
    assert np.all(w_np <= 5.0)
    assert np.all(w_np >= -5.0)

    x_th = torch.tensor(
        [float("nan"), float("inf"), float("-inf"), -2.0, 0.0, 2.0], dtype=torch.float64
    )
    y_th = _finite(x_th, max_abs=10.0)
    assert torch.isfinite(y_th).all()
    assert y_th[0].item() == 0.0
    assert y_th[1].item() == 10.0
    assert y_th[2].item() == -10.0

    z_th = _clamp_pos(torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64), eps=1e-3, max_abs=10.0)
    assert torch.all(z_th >= 1e-3)

    w_th = _clamp_any(torch.tensor([-20.0, 0.0, 20.0], dtype=torch.float64), max_abs=5.0)
    assert torch.all(w_th <= 5.0)
    assert torch.all(w_th >= -5.0)


def test_sanitize_state_clamps_positive_and_signed_sets():
    vals = {
        "x": np.array([np.nan, 2.0, -3.0]),
        "s": np.array([np.inf, -np.inf, 0.0]),
        "f": np.array([-1.0, 0.0, 2.0]),
        "v": np.array([0.0, -5.0, 1.0]),
        "q": np.array([-np.inf, 0.5, np.inf]),
        "v*": np.array([9999.0, -9999.0, 0.0]),
        "q*": None,
    }
    out = sanitize_state(vals, eps=1e-2, max_abs=10.0)

    # positive clamped
    assert np.all(out["f"] >= 1e-2)
    assert np.all(out["v"] >= 1e-2)
    assert np.all(out["q"] >= 1e-2)

    # signed clamped and finite
    assert np.isfinite(out["x"]).all()
    assert np.isfinite(out["s"]).all()
    assert np.max(out["v*"]) <= 10.0
    assert np.min(out["v*"]) >= -10.0

    # None preserved
    assert out["q*"] is None


# -----------------------------
# Derivatives: balloon + delay
# -----------------------------


def test_balloon_derivatives_basic_no_drain_numpy_scalar():
    c = _mk_consts()
    layer = _mk_layer(tau=2.0, x=1.0, s=0.1, f=1.2, v=1.1, q=0.9)

    d = balloon_derivatives(layer, c, order="exact")
    assert set(d.keys()) == {"ds_dt", "df_dt", "dv_dt", "dq_dt"}
    for v in d.values():
        assert np.isfinite(np.asarray(v)).all()


@pytest.mark.parametrize("order", ["linear", "quadratic"])
def test_balloon_derivatives_basic_no_drain_numpy_scalar_other_orders(order):
    c = _mk_consts()
    layer = _mk_layer(tau=2.0, x=1.0, s=0.1, f=1.2, v=1.1, q=0.9)

    d = balloon_derivatives(layer, c, order=order)
    assert set(d.keys()) == {"ds_dt", "df_dt", "dv_dt", "dq_dt"}
    for v in d.values():
        assert np.isfinite(np.asarray(v)).all()


def test_balloon_derivatives_all_orders_rest_at_zero_for_resting_state():
    # At x=0, s=0, f=1, v=1, q=1 the RHS of the dv/dq ODEs should vanish for
    # every implemented order -- this is a property of the ODE formulation
    # (each order's dv/dq is built from an "exact rest at zero" decomposition),
    # not something to assume without checking.
    c = _mk_consts()
    for order in ("exact", "linear", "quadratic"):
        layer = _mk_layer(tau=1.7, x=0.0, s=0.0, f=1.0, v=1.0, q=1.0)
        d = balloon_derivatives(layer, c, order=order)
        assert np.isclose(np.asarray(d["dv_dt"]), 0.0, atol=1e-12), order
        assert np.isclose(np.asarray(d["dq_dt"]), 0.0, atol=1e-12), order


def test_balloon_derivatives_with_drain_adds_terms_and_requires_stars():
    c = _mk_consts()

    lower = _mk_layer(
        depth=0,
        tau=1.0,
        x=0.0,
        s=0.0,
        f=1.0,
        v=1.0,
        q=1.0,
        v_star=np.array(0.5, dtype=np.float64),
        q_star=np.array(-0.25, dtype=np.float64),
    )
    upper = _mk_layer(
        depth=1, tau=2.0, x=0.0, s=0.0, f=1.0, v=1.0, q=1.0, lambda_d=0.8, drain_from=lower
    )

    d_no_drain = balloon_derivatives(
        _mk_layer(depth=1, tau=2.0, x=0.0, s=0.0, f=1.0, v=1.0, q=1.0), c, order="exact"
    )
    d_drain = balloon_derivatives(upper, c, order="exact")

    # dv_dt and dq_dt should differ due to coupling
    assert not np.isclose(np.asarray(d_drain["dv_dt"]), np.asarray(d_no_drain["dv_dt"]))
    assert not np.isclose(np.asarray(d_drain["dq_dt"]), np.asarray(d_no_drain["dq_dt"]))

    # If drain_from provided but stars missing => error
    lower_bad = _mk_layer(
        depth=0, tau=1.0, x=0.0, s=0.0, f=1.0, v=1.0, q=1.0, v_star=None, q_star=None
    )
    upper_bad = _mk_layer(
        depth=1, tau=2.0, x=0.0, s=0.0, f=1.0, v=1.0, q=1.0, lambda_d=0.8, drain_from=lower_bad
    )
    with pytest.raises(ValueError, match="v_star/q_star"):
        balloon_derivatives(upper_bad, c, order="exact")


def test_delay_filter_derivatives_requires_allocated_delayed_states():
    layer = _mk_layer(tau=1.0, v_star=None, q_star=None)
    with pytest.raises(ValueError, match="Delayed states are None"):
        delay_filter_derivatives(layer, tau_d=2.0)


def test_delay_filter_derivatives_outputs_keys_and_finite():
    layer = _mk_layer(
        tau=1.0,
        v=1.2,
        q=0.8,
        v_star=np.array(0.1, dtype=np.float64),
        q_star=np.array(-0.2, dtype=np.float64),
    )
    d = delay_filter_derivatives(layer, tau_d=2.0)
    assert set(d.keys()) == {"dv_star_dt", "dq_star_dt"}
    assert np.isfinite(np.asarray(d["dv_star_dt"])).all()
    assert np.isfinite(np.asarray(d["dq_star_dt"])).all()


def test_get_inversion_derivatives_union_of_keys():
    c = _mk_consts()
    layer = _mk_layer(
        tau=1.0,
        x=0.0,
        s=0.0,
        f=1.0,
        v=1.0,
        q=1.0,
        v_star=np.array(0.0, dtype=np.float64),
        q_star=np.array(0.0, dtype=np.float64),
    )
    d = get_inversion_derivatives(layer, c, tau_d=2.0, order="exact")
    assert set(d.keys()) == {"ds_dt", "df_dt", "dv_dt", "dq_dt", "dv_star_dt", "dq_star_dt"}


# -----------------------------
# RK4 step
# -----------------------------


def test_rk4_step_raises_when_derivative_missing():
    def dy_fn(y):
        return {"ds_dt": y["s"]}  # missing df_dt

    y0 = {"s": np.array(1.0), "f": np.array(2.0)}
    with pytest.raises(KeyError, match="Missing derivative"):
        rk4_step(
            y0,
            dy_fn,
            0.1,
            state_keys=("s", "f"),
            deriv_keys={"s": "ds_dt", "f": "df_dt"},
        )


def test_rk4_step_matches_exponential_decay_for_scalar():
    def dy_fn(y):
        return {"dy_dt": -y["y"]}

    y0 = {"y": np.array(1.0, dtype=np.float64)}
    dt = 0.2
    y1 = rk4_step(y0, dy_fn, dt, state_keys=("y",), deriv_keys={"y": "dy_dt"})
    assert np.isclose(y1["y"], np.exp(-dt), rtol=0.0, atol=5e-6)


# -----------------------------
# simulate_cortex
# -----------------------------


def test_simulate_cortex_shape_validation():
    c = _mk_consts()
    layers = [_mk_layer(depth=0, tau=1.0), _mk_layer(depth=1, tau=1.0)]
    x0 = np.zeros((5, 4, 3), dtype=np.float64)
    x1 = np.zeros((6, 4, 3), dtype=np.float64)  # mismatched T
    with pytest.raises(ValueError, match="shape"):
        simulate_cortex(layers, c, [x0, x1], dt=0.1, tau_d=1.0, order="exact")

    with pytest.raises(ValueError, match="must match"):
        simulate_cortex(layers, c, [x0], dt=0.1, tau_d=1.0, order="exact")


def test_simulate_cortex_outputs_expected_keys_numpy_without_delays():
    c = _mk_consts()
    layers = [
        _mk_layer(depth=0, tau=1.0),
        _mk_layer(depth=1, tau=1.0),
    ]
    x_inputs = [np.zeros((4, 5, 6), dtype=np.float64) for _ in layers]
    out = simulate_cortex(layers, c, x_inputs, dt=0.1, tau_d=1.0, order="exact")

    assert set(out.keys()) == {0, 1}
    for depth in (0, 1):
        d = out[depth]
        assert set(d.keys()) == {"x", "s", "f", "v", "q"}
        for k in d:
            assert d[k].shape == (4, 5, 6)
            assert np.isfinite(d[k]).all()


def test_simulate_cortex_outputs_include_delays_when_allocated():
    c = _mk_consts()
    layer = _mk_layer(
        depth=0,
        tau=1.0,
        v_star=np.zeros((3, 2), dtype=np.float64),
        q_star=np.zeros((3, 2), dtype=np.float64),
    )
    x_inputs = [np.zeros((5, 3, 2), dtype=np.float64)]
    out = simulate_cortex([layer], c, x_inputs, dt=0.1, tau_d=2.0, order="exact")

    d = out[0]
    assert "v*" in d and "q*" in d
    assert d["v*"].shape == (5, 3, 2)
    assert d["q*"].shape == (5, 3, 2)


def test_simulate_cortex_works_with_torch_inputs_and_preserves_dtype_device():
    c = _mk_consts()
    device = torch.device("cpu")
    x0 = torch.zeros((4, 3, 2), dtype=torch.float32, device=device)

    layer = _mk_layer(depth=0, tau=1.0)
    # overwrite initial state tensors to torch to keep operations torch-friendly
    layer.state.x = torch.zeros((3, 2), dtype=torch.float32, device=device)
    layer.state.s = torch.zeros((3, 2), dtype=torch.float32, device=device)
    layer.state.f = torch.ones((3, 2), dtype=torch.float32, device=device)
    layer.state.v = torch.ones((3, 2), dtype=torch.float32, device=device)
    layer.state.q = torch.ones((3, 2), dtype=torch.float32, device=device)

    out = simulate_cortex([layer], c, [x0], dt=0.1, tau_d=1.0, order="exact")
    d = out[0]
    assert isinstance(d["x"], torch.Tensor)
    assert d["x"].dtype == torch.float32
    assert d["x"].device == device
    assert d["x"].shape == (4, 3, 2)


# -----------------------------
# get_bold_from_state
# -----------------------------


def test_get_bold_from_state_formula_matches_definition_numpy():
    c = _mk_consts()
    acq = _mk_acq()
    st = _np_state_scalar(v=1.2, q=0.8)

    bold = get_bold_from_state(st, acq, c, params=None)

    # V0 * (k1*(1-q) + k2*(1-q/v) + k3*(1-v))
    expected = c.V0 * (
        acq.k1 * (1.0 - st["q"]) + acq.k2 * (1.0 - st["q"] / st["v"]) + acq.k3 * (1.0 - st["v"])
    )
    assert np.allclose(bold, expected)


def test_get_bold_from_state_applies_psf_when_configured():
    c = _mk_consts()
    acq = _mk_acq()
    T, H, W = 5, 21, 21

    v = np.ones((T, H, W), dtype=np.float64)
    q = np.ones((T, H, W), dtype=np.float64)

    # create a spatial impulse in q so bold has spatial structure
    q[:, H // 2, W // 2] = 0.5

    st = {"v": v, "q": q}
    psf = PointSpreadFunction(fwhm=3.0)
    cfg = BoldPostProcessingConfig(layer_psf={7: psf}, noise=None, noise_amplitude=0.0)

    b0 = get_bold_from_state(st, acq, c, layer_depth=7, params=None)
    b1 = get_bold_from_state(st, acq, c, layer_depth=7, params=cfg)

    # With PSF, the center impulse should spread (peak reduced)
    assert b1.shape == b0.shape
    assert b1[:, H // 2, W // 2].max() < b0[:, H // 2, W // 2].max()


def test_get_bold_from_state_noise_amplitude_priority_snr_overrides_models_overrides_direct():
    c = _mk_consts()
    acq = _mk_acq()

    T = 64
    v = np.ones((T,), dtype=np.float64)
    q = np.linspace(0.9, 1.1, T).astype(np.float64)  # ensure nonzero signal power
    st = {"v": v, "q": q}

    # base (no noise)
    b_clean = get_bold_from_state(st, acq, c, params=None)

    # direct amplitude (should add noise)
    cfg_amp = BoldPostProcessingConfig(noise=FakeNoise("white", seed=0), noise_amplitude=0.5)
    b_amp = get_bold_from_state(st, acq, c, params=cfg_amp)
    assert not np.allclose(b_amp, b_clean)

    # noise_models overrides noise_amplitude
    nm = NoiseModel.preset("3T", V=1.0, TR=2.0, S0=1.0)
    cfg_models = BoldPostProcessingConfig(
        noise=FakeNoise("white", seed=0),
        noise_amplitude=0.0,
        noise_models=[(nm, 2.0)],
    )
    b_models = get_bold_from_state(st, acq, c, params=cfg_models)
    assert not np.allclose(b_models, b_clean)

    # snr_db overrides both (and snr_db=np.inf => amplitude 0)
    cfg_snr_inf = BoldPostProcessingConfig(
        noise=FakeNoise("white", seed=0),
        noise_amplitude=999.0,
        noise_models=[(nm, 999.0)],
        snr_db=np.inf,
    )
    b_snr_inf = get_bold_from_state(st, acq, c, params=cfg_snr_inf)
    assert np.allclose(b_snr_inf, b_clean)


def test_get_bold_from_state_snr_db_sets_amplitude_from_signal_power_and_is_reproducible():
    c = _mk_consts()
    acq = _mk_acq()
    T = 128
    v = np.ones((T,), dtype=np.float64)
    q = np.sin(np.linspace(0, 2 * np.pi, T)).astype(np.float64) * 0.05 + 1.0
    st = {"v": v, "q": q}

    cfg = BoldPostProcessingConfig(noise=FakeNoise("white", seed=123), snr_db=10.0)
    b1 = get_bold_from_state(st, acq, c, params=cfg)
    b2 = get_bold_from_state(st, acq, c, params=cfg)
    assert np.allclose(b1, b2)  # deterministic due to seed


def test_get_bold_from_state_adds_noise_for_torch_tensor_and_keeps_dtype_device():
    c = _mk_consts()
    acq = _mk_acq()
    device = torch.device("cpu")

    T = 64
    v = torch.ones((T,), dtype=torch.float32, device=device)
    q = torch.linspace(0.95, 1.05, T, dtype=torch.float32, device=device)
    st = {"v": v, "q": q}

    cfg = BoldPostProcessingConfig(noise=FakeNoise("uniform", seed=0), noise_amplitude=0.1)
    b = get_bold_from_state(st, acq, c, params=cfg)

    assert isinstance(b, torch.Tensor)
    assert b.dtype == torch.float32
    assert b.device == device

    b_clean = get_bold_from_state(st, acq, c, params=None)
    assert not torch.allclose(b, b_clean)


def test_get_bold_from_state_snr_db_inf_matches_none_but_finite_high_snr_adds_tiny_noise():
    c = _mk_consts()
    acq = _mk_acq()
    T = 64
    v = np.ones((T,), dtype=np.float64)
    q = np.linspace(0.9, 1.1, T).astype(np.float64)
    st = {"v": v, "q": q}

    b_none = get_bold_from_state(st, acq, c, params=None)

    cfg_inf = BoldPostProcessingConfig(noise=FakeNoise("white", seed=0), snr_db=np.inf)
    b_inf = get_bold_from_state(st, acq, c, params=cfg_inf)
    assert np.array_equal(b_inf, b_none)  # np.inf special-case => amplitude 0.0, no noise added

    cfg_high = BoldPostProcessingConfig(noise=FakeNoise("white", seed=0), snr_db=200.0)
    b_high = get_bold_from_state(st, acq, c, params=cfg_high)
    # a very high (but finite) snr_db still adds *some* noise, unlike np.inf
    assert not np.array_equal(b_high, b_none)
    assert np.allclose(b_high, b_none, atol=1e-6)


def test_get_bold_from_state_noise_models_sums_scaled_noise_std_matches_manual_calc():
    c = _mk_consts()
    acq = _mk_acq()
    T = 64
    v = np.ones((T,), dtype=np.float64)
    q = np.linspace(0.9, 1.1, T).astype(np.float64)
    st = {"v": v, "q": q}

    nm1 = NoiseModel.preset("3T", V=1.0, TR=2.0, S0=1.0)
    nm2 = NoiseModel.preset("7T", V=1.0, TR=2.0, S0=2.0)
    scale1, scale2 = 1.5, 0.25
    expected_amp = nm1.noise_std * scale1 + nm2.noise_std * scale2

    noise = FakeNoise("white", seed=7)
    cfg = BoldPostProcessingConfig(
        noise=noise, noise_amplitude=999.0, noise_models=[(nm1, scale1), (nm2, scale2)]
    )
    b = get_bold_from_state(st, acq, c, params=cfg)
    b_clean = get_bold_from_state(st, acq, c, params=None)

    expected_noise = _generate_bold_noise(np.asarray(b_clean).shape, noise, expected_amp)
    assert np.allclose(b, b_clean + expected_noise)


def test_get_bold_from_state_no_noise_when_amplitude_zero_or_noise_none():
    c = _mk_consts()
    acq = _mk_acq()
    st = {"v": np.ones((32,), dtype=np.float64), "q": np.ones((32,), dtype=np.float64)}

    b_clean = get_bold_from_state(st, acq, c, params=None)

    cfg_amp0 = BoldPostProcessingConfig(noise=FakeNoise("white", seed=0), noise_amplitude=0.0)
    b_amp0 = get_bold_from_state(st, acq, c, params=cfg_amp0)
    assert np.allclose(b_amp0, b_clean)

    cfg_no_noise = BoldPostProcessingConfig(noise=None, noise_amplitude=1.0)
    b_no_noise = get_bold_from_state(st, acq, c, params=cfg_no_noise)
    assert np.allclose(b_no_noise, b_clean)


# -----------------------------
# simulate_cortex — regression / numerical stability
# -----------------------------


def test_simulate_cortex_nontrivial_input_stays_finite_and_physically_plausible():
    """Non-zero neural input produces finite outputs with physically valid states.

    Blood flow (f), volume (v), and deoxy-Hb content (q) must remain positive
    throughout the simulation, a basic physical constraint of the Balloon model.
    """
    np.random.seed(42)
    c = _mk_consts()
    T, H, W = 30, 4, 4
    dt = 0.01

    x0 = (np.random.randn(T, H, W) * 0.1).astype(np.float64)
    x1 = (np.random.randn(T, H, W) * 0.1).astype(np.float64)
    layers = [_mk_layer(depth=0, tau=1.0), _mk_layer(depth=1, tau=1.5)]
    out = simulate_cortex(layers, c, [x0, x1], dt=dt, tau_d=1.0, order="exact")

    for depth in (0, 1):
        d = out[depth]
        assert set(d.keys()) >= {"x", "s", "f", "v", "q"}, f"Missing keys in depth {depth}"
        for key in ("x", "s", "f", "v", "q"):
            assert d[key].shape == (T, H, W), f"Wrong shape for depth {depth} key {key}"
            assert np.isfinite(d[key]).all(), f"Non-finite values in depth {depth} key {key}"
        # Physical plausibility: f, v, q stay positive
        assert (d["f"] > 0).all(), f"f must be positive at depth {depth}"
        assert (d["v"] > 0).all(), f"v must be positive at depth {depth}"
        assert (d["q"] > 0).all(), f"q must be positive at depth {depth}"


def test_simulate_cortex_near_resting_state_for_small_input():
    """With a small constant neural drive, v and q should stay near resting (1.0).

    This is a numerical regression guard: a large deviation from 1.0 would indicate
    a change in integration step size, constants, or ODE formulation.
    """
    c = _mk_consts()
    T, H, W = 20, 3, 3
    # Constant small input — physiology should stay near rest
    x0 = np.full((T, H, W), 0.05, dtype=np.float64)
    layer = _mk_layer(depth=0, tau=1.0)
    out = simulate_cortex([layer], c, [x0], dt=0.01, tau_d=1.0, order="exact")

    v = out[0]["v"]
    q = out[0]["q"]
    # With a small, constant drive and short simulation, v and q should not
    # deviate more than 15 % from their resting value of 1.0.
    assert (
        np.abs(v.mean() - 1.0) < 0.15
    ), f"v mean {v.mean():.4f} deviated too far from resting state 1.0"
    assert (
        np.abs(q.mean() - 1.0) < 0.15
    ), f"q mean {q.mean():.4f} deviated too far from resting state 1.0"


def test_simulate_cortex_zero_input_stays_at_resting_state():
    """With zero neural input the Balloon model stays at its initial (resting) state.

    f, v, q are initialised to 1.0; s is 0.0.  Under zero forcing the RHS of every
    ODE evaluates to zero at these values, so the state should remain constant to
    within numerical noise from the RK4 integrator.
    """
    c = _mk_consts()
    T, H, W = 10, 3, 3
    x0 = np.zeros((T, H, W), dtype=np.float64)
    layer = _mk_layer(depth=0, tau=1.0)  # f=1, v=1, q=1, s=0 at init
    out = simulate_cortex([layer], c, [x0], dt=0.01, tau_d=1.0, order="exact")

    d = out[0]
    assert np.allclose(d["f"], 1.0, atol=1e-6), "f drifted from resting 1.0 under zero input"
    assert np.allclose(d["v"], 1.0, atol=1e-6), "v drifted from resting 1.0 under zero input"
    assert np.allclose(d["q"], 1.0, atol=1e-6), "q drifted from resting 1.0 under zero input"
    assert np.allclose(d["s"], 0.0, atol=1e-6), "s drifted from 0.0 under zero input"
