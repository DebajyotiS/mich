"""Pulse waveform and noise generators for synthetic neural-source signals.

Every `*Pulse` dataclass is a named forcing-signal shape: `generate(t)` returns
the pulse's amplitude at each time in `t` (zero outside its support), so all of
them share the interface `Pulse`'s factory (`_make_pulse`) dispatches over.
`ExpDecayPulse` and `RectPulse` are the two currently-recommended shapes;
`TriangularPulse`, `SincPulse`, and `AlphaPulse` are deprecated in favour of
those two (see each class's `@deprecated` reason).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np
from deprecated import deprecated


@dataclass(frozen=True, slots=True)
class ExpDecayPulse:
    """Exponential decay starting at `t_onset`: `amplitude * exp(-decay_rate *
    (t - t_onset))` for `t >= t_onset`, else 0."""

    amplitude: float
    t_onset: float
    decay_rate: float

    def generate(self, t: np.ndarray) -> np.ndarray:
        signal = np.zeros_like(t)
        mask = t >= self.t_onset
        signal[mask] += self.amplitude * np.exp(-self.decay_rate * (t[mask] - self.t_onset))
        return signal


@dataclass(frozen=True, slots=True)
class RectPulse:
    """Rectangular (boxcar) pulse: `amplitude` for `t` in `[t_onset, t_onset +
    width)`, else 0."""

    amplitude: float
    t_onset: float
    width: float

    def generate(self, t: np.ndarray) -> np.ndarray:
        signal = np.zeros_like(t)
        mask = (t >= self.t_onset) & (t < self.t_onset + self.width)
        signal[mask] += self.amplitude
        return signal


@deprecated(reason="TriangularPulse is deprecated. Use RectPulse or ExpDecayPulse instead.")
@dataclass(frozen=True, slots=True)
class TriangularPulse:
    """Symmetric triangular pulse of `width` centred at `t_peak`, linearly
    ramping 0 -> `amplitude` -> 0."""

    amplitude: float
    t_peak: float
    width: float

    def generate(self, t: np.ndarray) -> np.ndarray:
        signal = np.zeros_like(t)
        t_start = self.t_peak - self.width / 2
        t_end = self.t_peak + self.width / 2

        half_width = self.width / 2
        if half_width == 0:
            return signal  # avoid division by zero while preserving "no contribution" behavior

        mask_rise = (t >= t_start) & (t < self.t_peak)
        signal[mask_rise] += self.amplitude * (t[mask_rise] - t_start) / half_width

        mask_fall = (t >= self.t_peak) & (t < t_end)
        signal[mask_fall] += self.amplitude * (t_end - t[mask_fall]) / half_width
        return signal


@deprecated(reason="TriangularPulse is deprecated. Use RectPulse or ExpDecayPulse instead.")
@dataclass(frozen=True, slots=True)
class SincPulse:
    """Hamming-windowed sinc pulse of `width` centred at `t_center`, oscillating
    `cycles` full periods within that width."""

    amplitude: float
    t_center: float
    width: float
    cycles: float

    def generate(self, t: np.ndarray) -> np.ndarray:
        signal = np.zeros_like(t)
        t_start = self.t_center - self.width / 2
        t_end = self.t_center + self.width / 2
        mask = (t >= t_start) & (t < t_end)

        half_width = self.width / 2
        if half_width == 0:
            return signal  # avoid division by zero while preserving "no contribution" behavior

        t_norm = (t[mask] - self.t_center) / half_width
        sinc_arg = self.cycles * np.pi * t_norm
        sinc_val = np.sinc(sinc_arg / np.pi)
        window = 0.54 + 0.46 * np.cos(np.pi * t_norm)
        signal[mask] += self.amplitude * sinc_val * window
        return signal


@deprecated(reason="TriangularPulse is deprecated. Use RectPulse or ExpDecayPulse instead.")
@dataclass(frozen=True, slots=True)
class AlphaPulse:
    """Alpha-function pulse starting at `t_onset`: `amplitude * alpha * t' *
    exp(-beta * t')` where `t' = t - t_onset`, for `t >= t_onset`."""

    amplitude: float
    t_onset: float
    alpha: float
    beta: float

    def generate(self, t: np.ndarray) -> np.ndarray:
        signal = np.zeros_like(t)
        mask = t >= self.t_onset
        t_shifted = t[mask] - self.t_onset
        signal[mask] += self.amplitude * (self.alpha * t_shifted * np.exp(-self.beta * t_shifted))
        return signal


@dataclass(frozen=True, slots=True)
class Pulse:
    pulse_type: str
    peaks: list[list[float]]
    duration: float
    dt: float = 0.01
    baseline: Literal["fixed", "random"] = "fixed"
    rng: np.random.Generator | None = None

    def generate(self) -> tuple[np.ndarray, np.ndarray]:
        """Sum every peak's pulse waveform onto a shared `[0, duration)` time
        grid, then optionally inject a random baseline.

        Each entry of `peaks` is passed as positional args to the
        `pulse_type`-selected pulse dataclass (via `_make_pulse`), so its
        length/order must match that dataclass's fields.

        If `pulse_type == "rect"` and `baseline == "random"`, every
        *interior* zero-signal run (i.e. excluding the leading run before the
        first pulse and the trailing run after the last, via the `[1:-1]`
        slice below) gets an i.i.d. `Uniform(-0.1, 0.1) * median(peak
        amplitudes)` offset added across its whole span. TODO(doc): rationale
        unknown -- why those two boundary runs are excluded from baseline
        injection isn't stated.

        Returns:
            (t, signal), both `[N]` where `N = len(arange(0, duration, dt))`.
        """
        # NOTE: preserves old functionality: fixed dt=0.01 here
        t = np.arange(0, self.duration, self.dt)
        signal = np.zeros_like(t)

        for peak in self.peaks:
            pulse = _make_pulse(self.pulse_type, peak)
            signal += pulse.generate(t)

        if self.pulse_type == "rect" and self.baseline == "random":
            rng = self.rng if self.rng is not None else np.random.default_rng()
            mask = np.isclose(signal, 0.0, atol=1e-9).astype(int)
            padded = np.pad(mask, 1, mode="constant", constant_values=0)
            diffs = np.diff(padded)
            starts = np.where(diffs == 1)[0]
            ends = np.where(diffs == -1)[0] - 1
            zero_intervals = list(zip(starts, ends, strict=False))[1:-1]
            median_amplitude = np.median([peak[0] for peak in self.peaks])
            for start, end in zero_intervals:
                random_baseline = rng.uniform(-0.1 * median_amplitude, 0.1 * median_amplitude)
                signal[start : end + 1] += random_baseline
        return t, signal


def _make_pulse(pulse_type: str, peak: Sequence[Any]) -> Any:
    """Build the pulse dataclass named by `pulse_type`, from positional args `peak`.

    Args:
        pulse_type: One of "exp_decay", "rect", "gaussian" (maps to
            `TriangularPulse`, not a Gaussian pulse), "sinc", "alpha".
        peak: Positional constructor args for that pulse dataclass.

    Raises:
        ValueError: If `pulse_type` doesn't match one of the names above.
    """
    # Factory preserves same mapping from pulse_type to dataclass
    if pulse_type == "exp_decay":
        return ExpDecayPulse(*peak)
    if pulse_type == "rect":
        return RectPulse(*peak)
    if pulse_type == "gaussian":
        return TriangularPulse(*peak)
    if pulse_type == "sinc":
        return SincPulse(*peak)
    if pulse_type == "alpha":
        return AlphaPulse(*peak)
    raise ValueError(f"Unknown pulse type: {pulse_type}")


class Sources:
    """Accumulates source specs as plain dicts, in the `{"layer", "position",
    "signal"}` shape `LayeredDiffusionSimulator.simulate` expects for its
    `sources` argument."""

    def __init__(self) -> None:
        self.source_list: list[dict[str, Any]] = []

    def add_source(self, layer: int, position: tuple[int, int], signal: np.ndarray) -> None:
        """Append one source: `layer` index, `(h, w)` grid `position`, and its
        1-D `signal` timecourse."""
        # preserves old structure: dict with keys 'layer', 'position', 'signal'
        self.source_list.append({"layer": layer, "position": position, "signal": signal})

    def get_sources(self) -> list[dict[str, Any]]:
        """All sources added so far, in `add_source` order."""
        return self.source_list


NoiseDomain = Literal["spatial", "temporal", "both"]
NoiseType = Literal["white", "pink", "uniform"]


@dataclass(frozen=True, slots=True)
class Noise:
    """White/uniform/pink noise generator for either the spatial grid
    (`generate`) or per-source temporal traces (`generate_temporal`).

    `domain` records which of the two a caller should use (and, for "both",
    that both should be combined) -- neither `generate` nor `generate_temporal`
    reads `domain` itself; dispatching on it is the caller's responsibility
    (see `mich.data.neuronal.LayeredDiffusionSimulator.simulate`).
    """

    type: NoiseType
    seed: int | None = None
    domain: NoiseDomain = "spatial"

    def generate(self, amplitude: float, layers: int, grid_size: tuple[int, int]) -> np.ndarray:
        """Spatial noise field, one independent draw per layer.

        Pink noise is shaped per layer via `1/sqrt(|f|)` in the 2-D FFT domain
        (`|f|` = radial spatial frequency magnitude, DC bin left at 1.0 to
        avoid dividing by 0), then rescaled to zero-mean/unit-std before
        applying `amplitude` -- except when a layer's pink draw has near-zero
        variance (`std < 1e-12`), which returns exact zeros for that layer
        instead of amplifying near-nothing by dividing by a near-zero std.

        Args:
            amplitude: Target noise std (white/pink) or half-width (uniform);
                0 short-circuits to exact zeros without drawing any randomness.
            layers, grid_size: Output shape is `(layers, *grid_size)`.

        Raises:
            ValueError: If `self.type` isn't "white", "uniform", or "pink".
        """
        rng = np.random.default_rng(self.seed)

        if amplitude == 0.0:
            return np.zeros((layers, *grid_size), dtype=np.float64)

        if self.type == "white":
            return rng.normal(0.0, amplitude, size=(layers, *grid_size)).astype(np.float64)

        if self.type == "uniform":
            return rng.uniform(-amplitude, amplitude, size=(layers, *grid_size)).astype(np.float64)

        if self.type == "pink":
            noise = np.zeros((layers, *grid_size), dtype=np.float64)
            for layer in range(layers):
                white = rng.standard_normal(size=grid_size)
                white_fft = np.fft.fft2(white)

                freq_x = np.fft.fftfreq(grid_size[0])
                freq_y = np.fft.fftfreq(grid_size[1])
                fx, fy = np.meshgrid(freq_y, freq_x)
                fmag = np.sqrt(fx**2 + fy**2)
                fmag[0, 0] = 1.0

                pink_fft = white_fft / np.sqrt(fmag)
                pink = np.fft.ifft2(pink_fft).real

                std = np.std(pink)
                if std < 1e-12:
                    pink = np.zeros_like(pink)
                else:
                    pink = (pink - np.mean(pink)) / std

                noise[layer] = pink * amplitude
            return noise

        raise ValueError(f"Unknown noise type: {self.type}")

    def generate_temporal(
        self, amplitude: float, n_sources: int, steps: int, dt: float
    ) -> np.ndarray:
        """Temporal noise trace per source, one independent draw per source.

        Pink noise is shaped via `1/sqrt(freq)` in the 1-D (real) FFT domain
        (DC bin left at the first nonzero frequency to avoid dividing by 0),
        then rescaled to zero-mean/unit-std before applying `amplitude` --
        unlike `generate`'s spatial pink noise, there is no near-zero-std
        fallback here (a near-constant draw is divided by its own near-zero
        std as-is).

        Args:
            amplitude: Target noise std; 0 short-circuits to exact zeros.
            n_sources, steps: Output shape is `(n_sources, steps)`.
            dt: Sample spacing, used only for pink noise's frequency axis.

        Raises:
            ValueError: If `self.type` isn't "white", "uniform", or "pink".
        """
        rng = np.random.default_rng(self.seed)

        if amplitude == 0.0:
            return np.zeros((n_sources, steps), dtype=np.float64)

        if self.type == "white":
            return rng.normal(0.0, amplitude, size=(n_sources, steps)).astype(np.float64)

        if self.type == "uniform":
            return rng.uniform(-amplitude, amplitude, size=(n_sources, steps)).astype(np.float64)

        if self.type == "pink":
            # 1/f in time via FFT shaping per source
            out = np.zeros((n_sources, steps), dtype=np.float64)
            freqs = np.fft.rfftfreq(steps, d=dt)
            freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
            shaping = 1.0 / np.sqrt(freqs)  # amplitude shaping -> 1/f power

            for k in range(n_sources):
                x = rng.standard_normal(steps)
                X = np.fft.rfft(x)
                X *= shaping
                y = np.fft.irfft(X, n=steps)
                y = (y - y.mean()) / (y.std() + 1e-12)
                out[k] = amplitude * y
            return out

        raise ValueError(f"Unknown noise type: {self.type}")
