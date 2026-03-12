from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from src.data.neuronal import LayeredDiffusionSimulator, NeuralSimulatorParams

# -------------------------
# Helpers / test doubles
# -------------------------


class SpyNoise:
    """
    Minimal noise object to exercise routing and validation in simulate().
    Records calls and can be configured to return specific arrays.
    """

    def __init__(
        self,
        *,
        domain="spatial",
        temporal_fraction=0.5,
        spatial_return=None,
        temporal_return=None,
        spatial_nonfinite=False,
        temporal_nonfinite=False,
    ):
        self.domain = domain
        self.temporal_fraction = temporal_fraction
        self.calls = {"generate": 0, "generate_temporal": 0}
        self._spatial_return = spatial_return
        self._temporal_return = temporal_return
        self._spatial_nonfinite = spatial_nonfinite
        self._temporal_nonfinite = temporal_nonfinite

    def generate(self, amplitude: float, layers: int, grid_size: tuple[int, int]) -> np.ndarray:
        self.calls["generate"] += 1
        if self._spatial_return is not None:
            out = np.array(self._spatial_return, copy=True)
        else:
            out = np.zeros((layers, *grid_size), dtype=np.float64)
        out = out.astype(np.float64, copy=False)
        if self._spatial_nonfinite and out.size > 0:
            out.flat[0] = np.nan
        return out

    def generate_temporal(
        self, amplitude: float, n_sources: int, steps: int, dt: float
    ) -> np.ndarray:
        self.calls["generate_temporal"] += 1
        if self._temporal_return is not None:
            out = np.array(self._temporal_return, copy=True)
        else:
            out = np.zeros((n_sources, steps), dtype=np.float64)
        out = out.astype(np.float64, copy=False)
        if self._temporal_nonfinite and out.size > 0:
            out.flat[0] = np.inf
        return out


class NoTemporalNoise:
    """Noise object intentionally missing generate_temporal()."""

    def __init__(self, *, domain="temporal"):
        self.domain = domain

    def generate(self, amplitude: float, layers: int, grid_size: tuple[int, int]) -> np.ndarray:
        return np.zeros((layers, *grid_size), dtype=np.float64)


def _make_sim(
    *,
    num_layers=2,
    grid_size=(5, 6),
    dt=0.1,
    dx=1.0,
    diff_intra=0.0,
    diff_inter=0.0,
    safety=0.9,
    max_substeps=128,
    decay_rate=0.5,
):
    params = NeuralSimulatorParams(
        num_layers=num_layers,
        grid_size=grid_size,
        dt=dt,
        dx=dx,
        diffusion_coefficient_inter=diff_inter,
        diffusion_coefficient_intra=diff_intra,
        safety=safety,
        max_substeps=max_substeps,
        decay_rate=decay_rate,
    )
    return LayeredDiffusionSimulator(params)


def _source(layer=0, position=(1, 2), signal=None):
    if signal is None:
        signal = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    return {"layer": layer, "position": position, "signal": signal}


# -------------------------
# NeuralSimulatorParams
# -------------------------


def test_params_frozen_and_defaults():
    p = NeuralSimulatorParams(num_layers=1, grid_size=(3, 4))
    assert p.dt == 0.01
    assert p.dx == 1.0
    assert p.safety == 0.9
    assert p.max_substeps == 128
    assert p.noise_as_sde is True

    with pytest.raises((FrozenInstanceError, AttributeError)):
        p.dt = 0.02  # type: ignore[misc]


# -------------------------
# LayeredDiffusionSimulator.__init__
# -------------------------


def test_sim_init_wires_params_and_grid_shape():
    sim = _make_sim(num_layers=3, grid_size=(4, 5), dt=0.02, dx=2.0, diff_intra=0.1, diff_inter=0.2)
    assert sim.n_layers == 3
    assert sim.grid_size == (4, 5)
    assert sim.dt == 0.02
    assert sim.dx == 2.0
    assert sim.diff_intra == 0.1
    assert sim.diff_inter == 0.2
    assert sim.grid.shape == (3, 4, 5)
    assert np.all(sim.grid == 0.0)


def test_sim_init_max_substeps_matches_params():
    params = NeuralSimulatorParams(num_layers=1, grid_size=(3, 3), max_substeps=77)
    sim = LayeredDiffusionSimulator(params)
    assert sim.max_substeps == 77


# -------------------------
# generate_pulse wrapper
# -------------------------


def test_generate_pulse_forwards_to_pulse_generate(monkeypatch):
    sim = _make_sim()
    calls = {"n": 0}

    class DummyPulse:
        def generate(self):
            calls["n"] += 1
            t = np.array([0.0, 0.1], dtype=np.float64)
            y = np.array([1.0, 2.0], dtype=np.float64)
            return t, y

    out = sim.generate_pulse([DummyPulse(), DummyPulse()])
    assert calls["n"] == 2
    assert len(out) == 2
    assert np.allclose(out[0][0], [0.0, 0.1])
    assert np.allclose(out[0][1], [1.0, 2.0])


def test_generate_pulse_empty_list():
    sim = _make_sim()
    assert sim.generate_pulse([]) == []


# -------------------------
# simulate: steps validation
# -------------------------


@pytest.mark.parametrize("steps", [None, 0, -1])
def test_simulate_steps_must_be_positive(steps):
    sim = _make_sim()
    noise = SpyNoise(domain="spatial")
    with pytest.raises(ValueError, match="steps.*positive"):
        sim.simulate(_source(), steps=steps, snr_db=np.inf, noise=noise)  # type: ignore[arg-type]


# -------------------------
# simulate: sources validation
# -------------------------


def test_simulate_accepts_sources_as_dict_or_list_equivalently():
    sim = _make_sim(num_layers=1, grid_size=(4, 4), dt=0.1, dx=1.0, diff_intra=0.0, diff_inter=0.0)
    src = _source(layer=0, position=(2, 2), signal=np.array([1.0, 0.0, 0.0], dtype=np.float64))
    noise = SpyNoise(domain="spatial")

    h1 = sim.simulate(src, steps=3, snr_db=np.inf, noise=noise)
    # reset sim state by re-instantiating (simulate overwrites grid anyway, but keep clean)
    sim2 = _make_sim(num_layers=1, grid_size=(4, 4), dt=0.1, dx=1.0, diff_intra=0.0, diff_inter=0.0)
    noise2 = SpyNoise(domain="spatial")
    h2 = sim2.simulate([dict(src)], steps=3, snr_db=np.inf, noise=noise2)

    assert np.allclose(h1, h2)


@pytest.mark.parametrize("missing_key", ["layer", "position", "signal"])
def test_simulate_source_missing_keys_raises(missing_key):
    sim = _make_sim()
    src = _source()
    src.pop(missing_key)
    noise = SpyNoise(domain="spatial")
    with pytest.raises(ValueError, match="must have 'layer'.*'position'.*'signal'"):
        sim.simulate([src], steps=3, snr_db=np.inf, noise=noise)


def test_simulate_source_layer_out_of_bounds_raises():
    sim = _make_sim(num_layers=2)
    noise = SpyNoise(domain="spatial")
    with pytest.raises(ValueError, match="invalid layer"):
        sim.simulate([_source(layer=2)], steps=3, snr_db=np.inf, noise=noise)


def test_simulate_source_position_wrong_length_raises():
    sim = _make_sim()
    noise = SpyNoise(domain="spatial")
    with pytest.raises(ValueError, match="position must be"):
        sim.simulate(
            [{"layer": 0, "position": (1, 2, 3), "signal": np.ones(3)}],
            steps=3,
            snr_db=np.inf,
            noise=noise,
        )


def test_simulate_source_position_out_of_bounds_raises():
    sim = _make_sim(grid_size=(3, 3))
    noise = SpyNoise(domain="spatial")
    with pytest.raises(ValueError, match="position out of bounds"):
        sim.simulate([_source(layer=0, position=(10, 1))], steps=3, snr_db=np.inf, noise=noise)


def test_simulate_source_signal_must_be_1d():
    sim = _make_sim()
    noise = SpyNoise(domain="spatial")
    src = _source(signal=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="signal must be 1D"):
        sim.simulate([src], steps=3, snr_db=np.inf, noise=noise)


def test_simulate_source_signal_nonfinite_raises():
    sim = _make_sim()
    noise = SpyNoise(domain="spatial")
    sig = np.array([0.0, np.nan, 1.0])
    src = _source(signal=sig)
    with pytest.raises(ValueError, match="contains non-finite"):
        sim.simulate([src], steps=3, snr_db=np.inf, noise=noise)


def test_simulate_coerces_signal_to_float64_in_source_dict():
    sim = _make_sim()
    noise = SpyNoise(domain="spatial")
    sig = np.array([1, 2, 3], dtype=np.int32)
    src = _source(signal=sig)
    sim.simulate([src], steps=3, snr_db=np.inf, noise=noise)
    assert src["signal"].dtype == np.float64


# -------------------------
# simulate: parameter sanity checks
# -------------------------


def test_simulate_bad_dt_raises():
    sim = _make_sim(dt=0.1)
    sim.dt = 0.0  # mutate allowed (sim is not frozen)
    noise = SpyNoise(domain="spatial")
    with pytest.raises(ValueError, match="Bad dt"):
        sim.simulate([_source()], steps=3, snr_db=np.inf, noise=noise)


def test_simulate_bad_dx_raises():
    sim = _make_sim(dx=1.0)
    sim.dx = -1.0
    noise = SpyNoise(domain="spatial")
    with pytest.raises(ValueError, match="Bad dx"):
        sim.simulate([_source()], steps=3, snr_db=np.inf, noise=noise)


def test_simulate_negative_diffusion_raises():
    sim = _make_sim(diff_intra=0.0, diff_inter=0.0)
    sim.diff_intra = -0.1
    noise = SpyNoise(domain="spatial")
    with pytest.raises(ValueError, match="Diffusion coefficients must be non-negative"):
        sim.simulate([_source()], steps=3, snr_db=np.inf, noise=noise)


# -------------------------
# simulate: noise.domain routing + errors
# -------------------------


def test_simulate_unknown_noise_domain_raises():
    sim = _make_sim()
    noise = SpyNoise(domain="weird")
    with pytest.raises(ValueError, match="Unknown noise.domain"):
        sim.simulate([_source()], steps=3, snr_db=0.0, noise=noise)


def test_simulate_temporal_domain_requires_generate_temporal():
    sim = _make_sim()
    noise = NoTemporalNoise(domain="temporal")
    with pytest.raises(AttributeError, match="requests temporal noise"):
        sim.simulate([_source()], steps=3, snr_db=0.0, noise=noise)


def test_simulate_temporal_noise_shape_mismatch_raises():
    sim = _make_sim()
    # wrong shape: should be (n_sources, steps)
    noise = SpyNoise(domain="temporal", temporal_return=np.zeros((1, 999), dtype=np.float64))
    with pytest.raises(ValueError, match="Temporal noise has shape"):
        sim.simulate([_source()], steps=3, snr_db=0.0, noise=noise)


def test_simulate_temporal_noise_nonfinite_raises():
    sim = _make_sim()
    noise = SpyNoise(domain="temporal", temporal_nonfinite=True)
    with pytest.raises(ValueError, match="returned non-finite"):
        sim.simulate([_source()], steps=3, snr_db=0.0, noise=noise)


def test_simulate_spatial_noise_shape_mismatch_raises():
    sim = _make_sim(num_layers=2, grid_size=(3, 3))
    # wrong shape returned by generate
    noise = SpyNoise(domain="spatial", spatial_return=np.zeros((1, 3, 3)))
    with pytest.raises(ValueError, match="Noise grid has shape"):
        sim.simulate([_source()], steps=3, snr_db=0.0, noise=noise)


def test_simulate_spatial_noise_nonfinite_raises():
    sim = _make_sim()
    noise = SpyNoise(domain="spatial", spatial_nonfinite=True)
    with pytest.raises(ValueError, match="returned non-finite"):
        sim.simulate([_source()], steps=3, snr_db=0.0, noise=noise)


def test_simulate_snr_inf_disables_noise_generators():
    sim = _make_sim()
    noise = SpyNoise(domain="both")
    sim.simulate([_source()], steps=3, snr_db=np.inf, noise=noise)
    # Pn_total=0 -> amplitudes 0 -> should not call either generator
    assert noise.calls["generate"] == 0
    assert noise.calls["generate_temporal"] == 0


def test_simulate_domain_spatial_calls_only_generate_when_noise_needed():
    sim = _make_sim()
    noise = SpyNoise(domain="spatial")
    sim.simulate([_source()], steps=3, snr_db=0.0, noise=noise)
    assert noise.calls["generate"] > 0
    assert noise.calls["generate_temporal"] == 0


def test_simulate_domain_temporal_calls_only_generate_temporal_when_noise_needed():
    sim = _make_sim()
    noise = SpyNoise(domain="temporal")
    sim.simulate([_source()], steps=3, snr_db=0.0, noise=noise)
    assert noise.calls["generate"] == 0
    assert noise.calls["generate_temporal"] == 1  # computed once up front


def test_simulate_domain_both_calls_both_when_fraction_in_range():
    sim = _make_sim()
    noise = SpyNoise(domain="both", temporal_fraction=0.4)
    sim.simulate([_source()], steps=3, snr_db=0.0, noise=noise)
    assert noise.calls["generate_temporal"] == 1
    assert noise.calls["generate"] > 0


def test_simulate_domain_both_fraction_clipped_to_zero_disables_temporal():
    sim = _make_sim()
    noise = SpyNoise(domain="both", temporal_fraction=-123.0)
    sim.simulate([_source()], steps=3, snr_db=0.0, noise=noise)
    assert noise.calls["generate_temporal"] == 0
    assert noise.calls["generate"] > 0


def test_simulate_domain_both_fraction_clipped_to_one_disables_spatial():
    sim = _make_sim()
    noise = SpyNoise(domain="both", temporal_fraction=999.0)
    sim.simulate([_source()], steps=3, snr_db=0.0, noise=noise)
    assert noise.calls["generate_temporal"] == 1
    assert noise.calls["generate"] == 0


# -------------------------
# simulate: stability / substepping
# -------------------------


def test_simulate_raises_when_explicit_diffusion_unstable_and_requires_too_many_substeps():
    # Make dt very large and diffusion large so required substeps is huge.
    sim = _make_sim(
        num_layers=1, grid_size=(5, 5), dt=10.0, dx=1.0, diff_intra=10.0, diff_inter=10.0
    )
    sim.safety = 0.9
    sim.max_substeps = 2  # override regardless of __init__ bug
    noise = SpyNoise(domain="spatial")
    with pytest.raises(ValueError, match="Unstable explicit diffusion"):
        sim.simulate([_source(signal=np.ones(3))], steps=3, snr_db=np.inf, noise=noise)


def test_simulate_runs_with_substepping_when_required_within_cap():
    # Construct a case where dt exceeds safety*dt_max but required is small.
    sim = _make_sim(
        num_layers=1, grid_size=(5, 5), dt=0.5, dx=1.0, diff_intra=1.0, diff_inter=0.0, safety=0.9
    )
    sim.max_substeps = 10  # override regardless of __init__ bug
    noise = SpyNoise(domain="spatial")
    h = sim.simulate([_source(signal=np.ones(5))], steps=5, snr_db=np.inf, noise=noise)
    assert h.shape == (5, 1, 5, 5)
    assert np.isfinite(h).all()


# -------------------------
# simulate: core dynamics sanity (no diffusion, no noise)
# -------------------------


def test_simulate_injection_sets_value_without_diffusion_or_noise():
    # With diff=0, decay=0, and snr=inf, injection assigns sig[step] to the grid point.
    # No accumulation -- the value at (layer, i, j) is exactly sig[step] after each step.
    sim = _make_sim(
        num_layers=1, grid_size=(4, 4), dt=0.2, dx=1.0, diff_intra=0.0, diff_inter=0.0, decay_rate=0.0
    )
    sig = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    src = _source(layer=0, position=(1, 1), signal=sig)
    noise = SpyNoise(domain="spatial")

    hist = sim.simulate([src], steps=4, snr_db=np.inf, noise=noise)
    assert hist.shape == (4, 1, 4, 4)

    # Injection point tracks signal exactly
    got = hist[:, 0, 1, 1]
    assert np.allclose(got, sig)

    # Everything else stays zero
    mask = np.ones((4, 1, 4, 4), dtype=bool)
    mask[:, 0, 1, 1] = False
    assert np.all(hist[mask] == 0.0)
