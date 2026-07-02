import numpy as np
import pytest

from src.data.signals import (
    AlphaPulse,
    ExpDecayPulse,
    TriangularPulse,
    Noise,
    Pulse,
    RectPulse,
    SincPulse,
    Sources,
    _make_pulse,
)


def _t_grid(duration=1.0, dt=0.1):
    return np.arange(0.0, duration, dt, dtype=np.float64)


# -----------------------
# Pulse dataclasses
# -----------------------


def test_exp_decay_pulse_generate_basic():
    t = _t_grid(1.0, 0.1)
    p = ExpDecayPulse(amplitude=2.0, t_onset=0.3, decay_rate=1.5)
    y = p.generate(t)

    assert y.shape == t.shape
    assert np.all(y[t < 0.3] == 0.0)

    idx = np.where(t >= 0.3)[0][0]
    # At onset: exp(0)=1 -> amplitude
    assert np.isclose(y[idx], 2.0)

    # Monotone decreasing after onset for positive decay_rate (on our grid)
    post = y[t >= 0.3]
    assert np.all(np.diff(post) <= 1e-12)


def test_rect_pulse_generate_basic():
    t = _t_grid(1.0, 0.1)
    p = RectPulse(amplitude=3.0, t_start=0.2, width=0.3)
    y = p.generate(t)

    assert y.shape == t.shape
    assert np.all(y[(t < 0.2) | (t >= 0.5)] == 0.0)
    assert np.all(y[(t >= 0.2) & (t < 0.5)] == 3.0)


def test_gaussian_pulse_generate_triangle_shape_and_zero_width_returns_zero():
    t = _t_grid(1.0, 0.1)

    # width=0 should return all zeros (division-by-zero guard path)
    p0 = TriangularPulse(amplitude=1.0, t_peak=0.5, width=0.0)
    y0 = p0.generate(t)
    assert np.all(y0 == 0.0)

    # nonzero width: piecewise linear ramp up/down
    p = TriangularPulse(amplitude=2.0, t_peak=0.5, width=0.4)
    y = p.generate(t)

    t_start = 0.5 - 0.2
    t_end = 0.5 + 0.2
    assert np.all(y[(t < t_start) | (t >= t_end)] == 0.0)

    # At exact peak (t == t_peak) included in "fall" mask, should be amplitude
    peak_idx = np.where(np.isclose(t, 0.5))[0]
    if len(peak_idx) > 0:
        assert np.isclose(y[peak_idx[0]], 2.0)

    # Values should be nonnegative and not exceed amplitude
    assert np.all(y >= -1e-12)
    assert np.max(y) <= 2.0 + 1e-12


def test_sinc_pulse_generate_windowed_and_zero_width_returns_zero():
    t = _t_grid(1.0, 0.001)

    # width=0 should return zeros (division-by-zero guard path)
    p0 = SincPulse(amplitude=1.0, t_center=0.5, width=0.0, cycles=3.0)
    y0 = p0.generate(t)
    assert np.all(y0 == 0.0)

    p = SincPulse(amplitude=2.0, t_center=0.5, width=0.2, cycles=2.0)
    y = p.generate(t)

    t_start = 0.5 - 0.1
    t_end = 0.5 + 0.1
    assert np.all(y[(t < t_start) | (t >= t_end)] == 0.0)

    # At center: t_norm=0 -> sinc=1, window=0.54+0.46*cos(0)=1 => amplitude
    center_idx = np.argmin(np.abs(t - 0.5))
    assert np.isclose(y[center_idx], 2.0, atol=1e-2)  # grid approx


def test_alpha_pulse_generate_basic():
    t = _t_grid(1.0, 0.1)
    p = AlphaPulse(amplitude=1.5, t_onset=0.2, alpha=2.0, beta=3.0)
    y = p.generate(t)

    assert np.all(y[t < 0.2] == 0.0)
    # At onset t_shifted=0 => contribution 0
    onset_idx = np.where(t >= 0.2)[0][0]
    assert np.isclose(y[onset_idx], 0.0)

    # Should rise above 0 shortly after onset (for positive alpha, beta)
    assert np.any(y[t > 0.2] > 0.0)


# -----------------------
# Factory and Pulse wrapper
# -----------------------


@pytest.mark.parametrize(
    "pulse_type, peak, cls",
    [
        ("exp_decay", [1.0, 0.2, 2.0], ExpDecayPulse),
        ("rect", [1.0, 0.2, 0.3], RectPulse),
        ("gaussian", [1.0, 0.5, 0.4], TriangularPulse),
        ("sinc", [1.0, 0.5, 0.4, 3.0], SincPulse),
        ("alpha", [1.0, 0.2, 2.0, 3.0], AlphaPulse),
    ],
)
def test_make_pulse_returns_correct_type(pulse_type, peak, cls):
    p = _make_pulse(pulse_type, peak)
    assert isinstance(p, cls)


def test_make_pulse_unknown_raises():
    with pytest.raises(ValueError, match="Unknown pulse type"):
        _make_pulse("does_not_exist", [1.0])


def test_pulse_generate_sums_multiple_peaks_and_respects_dt():
    # rect pulses are easiest to check exactly
    peaks = [
        [1.0, 0.2, 0.3],  # active in [0.2, 0.5)
        [2.0, 0.4, 0.2],  # active in [0.4, 0.6)
    ]
    P = Pulse(pulse_type="rect", peaks=peaks, duration=1.0, dt=0.1)
    t, y = P.generate()

    assert np.isclose(t[1] - t[0], 0.1)
    assert y.shape == t.shape

    # Regions:
    # [0.0,0.2): 0
    assert np.all(y[t < 0.2] == 0.0)
    # [0.2,0.4): 1
    assert np.all(y[(t >= 0.2) & (t < 0.4)] == 1.0)
    # [0.4,0.5): overlap -> 3
    assert np.all(y[(t >= 0.4) & (t < 0.5)] == 3.0)
    # [0.5,0.6): only second -> 2
    assert np.all(y[(t >= 0.5) & (t < 0.6)] == 2.0)
    # [0.6,1.0): 0
    assert np.all(y[t >= 0.6] == 0.0)


# -----------------------
# Sources
# -----------------------


def test_sources_add_and_get_sources_roundtrip():
    s = Sources()
    assert s.get_sources() == []

    sig = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    s.add_source(layer=1, position=(10, 20), signal=sig)

    out = s.get_sources()
    assert isinstance(out, list)
    assert len(out) == 1
    assert out[0]["layer"] == 1
    assert out[0]["position"] == (10, 20)
    assert out[0]["signal"] is sig  # preserves object reference per implementation


# -----------------------
# Noise: spatial
# -----------------------


def test_noise_generate_amplitude_zero_returns_zeros():
    n = Noise(type="white", seed=123, domain="spatial")
    y = n.generate(amplitude=0.0, layers=2, grid_size=(4, 5))

    assert y.shape == (2, 4, 5)
    assert y.dtype == np.float64
    assert np.all(y == 0.0)


@pytest.mark.parametrize("noise_type", ["white", "uniform"])
def test_noise_generate_reproducible_with_seed(noise_type):
    n1 = Noise(type=noise_type, seed=123, domain="spatial")
    n2 = Noise(type=noise_type, seed=123, domain="spatial")

    y1 = n1.generate(amplitude=1.0, layers=3, grid_size=(8, 8))
    y2 = n2.generate(amplitude=1.0, layers=3, grid_size=(8, 8))

    assert y1.shape == (3, 8, 8)
    assert y1.dtype == np.float64
    assert np.allclose(y1, y2)


def test_noise_generate_uniform_range():
    n = Noise(type="uniform", seed=0, domain="spatial")
    amp = 0.7
    y = n.generate(amplitude=amp, layers=2, grid_size=(16, 16))

    assert np.max(y) <= amp + 1e-12
    assert np.min(y) >= -amp - 1e-12


def test_noise_generate_white_has_nonzero_variance():
    n = Noise(type="white", seed=0, domain="spatial")
    y = n.generate(amplitude=1.0, layers=1, grid_size=(64, 64))

    assert np.var(y) > 1e-3


def test_noise_generate_pink_properties_and_reproducible():
    n1 = Noise(type="pink", seed=42, domain="spatial")
    n2 = Noise(type="pink", seed=42, domain="spatial")

    y1 = n1.generate(amplitude=1.0, layers=2, grid_size=(32, 32))
    y2 = n2.generate(amplitude=1.0, layers=2, grid_size=(32, 32))

    assert y1.shape == (2, 32, 32)
    assert y1.dtype == np.float64
    assert np.allclose(y1, y2)

    # pink branch normalizes to ~zero mean per layer before scaling by amplitude
    # allow slack because discretization and numerical effects exist
    assert np.all(np.abs(y1.mean(axis=(1, 2))) < 1e-1)
    assert np.all(np.std(y1, axis=(1, 2)) > 1e-2)


def test_noise_generate_unknown_type_raises():
    _n = Noise(type="white", seed=0, domain="spatial")
    # mutate by constructing invalid via object.__setattr__ is blocked by frozen dataclass;
    # so instantiate using a cast-like ignore: just pass a wrong literal at runtime.
    n_bad = Noise(type="not_a_type", seed=0, domain="spatial")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown noise type"):
        n_bad.generate(amplitude=1.0, layers=1, grid_size=(4, 4))


# -----------------------
# Noise: temporal
# -----------------------


def test_noise_generate_temporal_amplitude_zero_returns_zeros():
    n = Noise(type="white", seed=123, domain="temporal")
    y = n.generate_temporal(amplitude=0.0, n_sources=3, steps=10, dt=0.01)

    assert y.shape == (3, 10)
    assert y.dtype == np.float64
    assert np.all(y == 0.0)


@pytest.mark.parametrize("noise_type", ["white", "uniform"])
def test_noise_generate_temporal_reproducible_with_seed(noise_type):
    n1 = Noise(type=noise_type, seed=123, domain="temporal")
    n2 = Noise(type=noise_type, seed=123, domain="temporal")

    y1 = n1.generate_temporal(amplitude=1.0, n_sources=2, steps=128, dt=0.01)
    y2 = n2.generate_temporal(amplitude=1.0, n_sources=2, steps=128, dt=0.01)

    assert y1.shape == (2, 128)
    assert y1.dtype == np.float64
    assert np.allclose(y1, y2)


def test_noise_generate_temporal_uniform_range():
    n = Noise(type="uniform", seed=0, domain="temporal")
    amp = 0.3
    y = n.generate_temporal(amplitude=amp, n_sources=4, steps=50, dt=0.02)

    assert np.max(y) <= amp + 1e-12
    assert np.min(y) >= -amp - 1e-12


def test_noise_generate_temporal_pink_shape_and_stats():
    n = Noise(type="pink", seed=7, domain="temporal")
    amp = 0.8
    y = n.generate_temporal(amplitude=amp, n_sources=3, steps=256, dt=0.01)

    assert y.shape == (3, 256)
    assert y.dtype == np.float64

    # Each source is normalized then scaled by amplitude, so mean should be near 0
    assert np.all(np.abs(y.mean(axis=1)) < 1e-1)
    # Std should be near amplitude (since normalized to ~1 before scaling)
    assert np.all(np.abs(y.std(axis=1) - amp) < 2e-1)


def test_noise_generate_temporal_pink_steps_1_edge_case():
    # exercises: freqs length == 1 branch
    n = Noise(type="pink", seed=0, domain="temporal")
    y = n.generate_temporal(amplitude=1.0, n_sources=2, steps=1, dt=0.01)

    assert y.shape == (2, 1)
    # After normalization with (std + 1e-12) it should be finite
    assert np.all(np.isfinite(y))


def test_noise_generate_temporal_unknown_type_raises():
    n_bad = Noise(type="nope", seed=0, domain="temporal")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown noise type"):
        n_bad.generate_temporal(amplitude=1.0, n_sources=1, steps=16, dt=0.01)
