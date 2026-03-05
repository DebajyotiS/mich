from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class ExpDecayPulse:
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
    amplitude: float
    t_start: float
    width: float

    def generate(self, t: np.ndarray) -> np.ndarray:
        signal = np.zeros_like(t)
        mask = (t >= self.t_start) & (t < self.t_start + self.width)
        signal[mask] += self.amplitude
        return signal


@dataclass(frozen=True, slots=True)
class GaussianPulse:
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


@dataclass(frozen=True, slots=True)
class SincPulse:
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


@dataclass(frozen=True, slots=True)
class AlphaPulse:
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

    def generate(self) -> tuple[np.ndarray, np.ndarray]:
        # NOTE: preserves old functionality: fixed dt=0.01 here
        t = np.arange(0, self.duration, self.dt)
        signal = np.zeros_like(t)

        for peak in self.peaks:
            pulse = _make_pulse(self.pulse_type, peak)
            signal += pulse.generate(t)

        return t, signal


def _make_pulse(pulse_type: str, peak: Sequence[Any]) -> Any:
    # Factory preserves same mapping from pulse_type to dataclass
    if pulse_type == "exp_decay":
        return ExpDecayPulse(*peak)
    if pulse_type == "rect":
        return RectPulse(*peak)
    if pulse_type == "gaussian":
        return GaussianPulse(*peak)
    if pulse_type == "sinc":
        return SincPulse(*peak)
    if pulse_type == "alpha":
        return AlphaPulse(*peak)
    raise ValueError(f"Unknown pulse type: {pulse_type}")


class Sources:
    def __init__(self) -> None:
        self.source_list: list[dict[str, Any]] = []

    def add_source(self, layer: int, position: tuple[int, int], signal: np.ndarray) -> None:
        # preserves old structure: dict with keys 'layer', 'position', 'signal'
        self.source_list.append({"layer": layer, "position": position, "signal": signal})

    def get_sources(self) -> list[dict[str, Any]]:
        return self.source_list


NoiseDomain = Literal["spatial", "temporal", "both"]
NoiseType = Literal["white", "pink", "uniform"]


@dataclass(frozen=True, slots=True)
class Noise:
    type: NoiseType
    seed: int | None = None
    domain: NoiseDomain = "spatial"

    def generate(self, amplitude: float, layers: int, grid_size: tuple[int, int]) -> np.ndarray:
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
