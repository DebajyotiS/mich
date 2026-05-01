from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, TypeAlias

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from src.data.signals import Noise

Timecourse: TypeAlias = np.ndarray | torch.Tensor


@dataclass(frozen=True, slots=True)
class HaemodynamicConstants:
    kappa: float  # vasodilation rate constant
    gamma: float  # autoregulation rate constant
    alpha: float  # Grubb's exponent
    E0: float  # resting oxygen extraction fraction
    V0: float  # resting blood volume (used in BOLD readout, not in RHS)


@dataclass(frozen=True, slots=True)
class AcquisitionConstants:
    k1: float
    k2: float
    k3: float


@dataclass(slots=True)
class HaemodynamicState:
    x: Timecourse  # neural drive (input/forcing, not integrated)
    s: Timecourse  # vasodilatory signal
    f: Timecourse  # blood inflow
    v: Timecourse  # blood volume
    q: Timecourse  # deoxyhemoglobin content

    v_star: Timecourse | None = None  # delayed deviation (signed)
    q_star: Timecourse | None = None  # delayed deviation (signed)

    def as_dict(self) -> dict[str, Timecourse | None]:
        return {
            "x": self.x,
            "s": self.s,
            "f": self.f,
            "v": self.v,
            "q": self.q,
            "v*": self.v_star,
            "q*": self.q_star,
        }


@dataclass(slots=True)
class CortexLayer:
    depth: int
    tau: float  # transit time
    state: HaemodynamicState
    lambda_d: float = 0.0  # coupling strength
    drain_from: "CortexLayer | None" = None  # lower layer feeding this one


@dataclass(frozen=True, slots=True)
class PointSpreadFunction:
    """Gaussian point spread function for spatial blurring of BOLD signal.
    The FWHM (full-width at half-maximum) is specified in grid units and
    is typically different per cortical layer due to vascular drainage effects.
    """

    fwhm: float  # full-width at half-maximum in grid units
    boundary: str = "absorbing"  # "absorbing" (zero-pad) or "reflect"

    @property
    def sigma(self) -> float:
        """Gaussian sigma corresponding to this FWHM."""
        return self.fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    def kernel_2d(self) -> np.ndarray:
        """2-D separable Gaussian kernel as a (1, 1, k, k) float32 numpy array."""
        if self.fwhm <= 0.0:
            return np.array([[[[1.0]]]], dtype=np.float32)
        k1d = _gaussian_kernel_1d(self.sigma)
        k2d = (k1d[:, None] * k1d[None, :]).astype(np.float32)
        return k2d[np.newaxis, np.newaxis]  # (1, 1, k, k)

    def apply(self, data: Timecourse) -> Timecourse:
        """Apply Gaussian blur to spatial dimensions of data shaped (T, *spatial)."""
        if self.fwhm <= 0.0:
            return data
        sigma = self.sigma
        if isinstance(data, torch.Tensor):
            return _apply_psf_torch(data, sigma, boundary=self.boundary)
        return _apply_psf_numpy(data, sigma, boundary=self.boundary)


@dataclass(frozen=True, slots=True)
class NoiseModel:
    """fMRI noise model based on Triantafyllou et al. 2005.

    Decomposes noise into thermal and physiological contributions.
    The computed sigma is noise std relative to equilibrium signal S0.

    Parameters
    ----------
    kappa : float
        Thermal SNR parameter (field-strength dependent).
    lam : float
        Physiological noise fraction (lambda in the paper).
    T1 : float
        Longitudinal relaxation time [s].
    V : float
        Voxel volume [mm^3].
    TR : float
        Repetition time [s].
    nT : int
        Number of volumes averaged (scales thermal noise).
    differential : bool
        If True, nT/2 volumes per condition (A vs B subtraction).
    S0 : float
        Equilibrium signal level in BOLD-output units. ``sigma`` is noise
        std / S0, so ``noise_std = sigma * S0`` gives the absolute noise
        amplitude in the same units as the BOLD readout. Default 1.0
        (noise amplitude equals the dimensionless sigma).
    """

    kappa: float
    lam: float
    T1: float
    V: float
    TR: float
    nT: int = 1
    differential: bool = False
    S0: float = 1.0

    @classmethod
    def preset(
        cls,
        field: str,
        *,
        V: float,
        TR: float,
        nT: int = 1,
        differential: bool = False,
        S0: float = 1.0,
    ) -> "NoiseModel":
        presets = {
            "3T": dict(kappa=6.6567, lam=0.0129, T1=1.607),
            "7T": dict(kappa=9.9632, lam=0.0113, T1=1.939),
            "thermal": dict(kappa=9.9632, lam=0.0, T1=1.939),
            "physiological": dict(kappa=np.inf, lam=0.0113, T1=1.939),
        }
        if field not in presets:
            raise ValueError(f"Unknown preset '{field}'. Choose from {list(presets)}")
        return cls(**presets[field], V=V, TR=TR, nT=nT, differential=differential, S0=S0)

    @property
    def sigma(self) -> float:
        """Noise std relative to equilibrium signal S0."""
        TR0 = 5.4
        F = np.sqrt(np.tanh(self.TR / (2 * self.T1)) / np.tanh(TR0 / (2 * self.T1)))
        k = self.kappa * F

        if not self.differential and self.nT != 1:
            raise ValueError("Multiple measurements only supported with differential=True")

        if self.nT == 1:
            return float(np.sqrt(1 + self.lam**2 * k**2 * self.V**2) / (k * self.V))

        # differential case: AR(1) temporal correlation (tau=15s)
        half = self.nT // 2
        s = 0.0
        for t1 in range(1, half + 1):
            for t2 in range(1, half + 1):
                s += np.exp(-self.TR * abs(t1 - t2) / 15.0)
        return float(np.sqrt(4 / (k**2 * self.V**2 * self.nT) + (2 * self.lam**2) / half**2 * s))

    @property
    def noise_std(self) -> float:
        """Absolute noise std in BOLD-output units (sigma * S0)."""
        return self.sigma * self.S0


@dataclass(frozen=True, slots=True)
class BoldPostProcessingConfig:
    """Optional post-processing applied to the BOLD readout.

    Processing order: raw BOLD -> PSF convolution -> additive noise.

    Noise amplitude is determined by (in priority order):
    1. ``snr_db`` -- amplitude derived from signal power (highest priority)
    2. ``noise_models`` -- sum of ``noise_std * scale`` over each ``(NoiseModel, scale)`` pair
    3. ``noise_amplitude`` as a direct override
    """

    layer_psf: dict[int, PointSpreadFunction] | None = None  # depth -> PSF
    noise: Noise | None = None
    noise_amplitude: float = 0.0
    noise_models: list[tuple[NoiseModel, float]] | None = None
    snr_db: float | None = None


def _gaussian_kernel_1d(sigma: float, truncate: float = 4.0) -> np.ndarray:
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def _apply_psf_numpy(data: np.ndarray, sigma: float, *, boundary: str = "absorbing") -> np.ndarray:
    if data.ndim <= 1:
        return data
    # sigma=0 on the time axis, blur only spatial axes
    spatial_sigma = [0.0] + [sigma] * (data.ndim - 1)
    if boundary == "absorbing":
        return gaussian_filter(data, sigma=spatial_sigma, mode="constant", cval=0.0)
    return gaussian_filter(data, sigma=spatial_sigma, mode="reflect")


def _reflect_pad(x: torch.Tensor, pad: int, dim: int) -> torch.Tensor:
    """Reflect-pad tensor along `dim` by `pad` elements; works for pad >= size."""
    size = x.shape[dim]
    period = 2 * (size - 1)
    i = torch.arange(-pad, size + pad, device=x.device, dtype=torch.long)
    i = ((i % period) + period) % period
    idx = torch.where(i < size, i, period - i)
    return x.index_select(dim, idx)


def _apply_psf_torch(data: torch.Tensor, sigma: float, *, boundary: str = "absorbing") -> torch.Tensor:
    if data.ndim <= 1:
        return data
    kernel_np = _gaussian_kernel_1d(sigma)
    kernel = torch.as_tensor(kernel_np, dtype=data.dtype, device=data.device)
    pad = len(kernel_np) // 2
    n_spatial = data.ndim - 1

    if n_spatial == 1:
        k = kernel.reshape(1, 1, -1)
        if boundary == "absorbing":
            return torch.nn.functional.conv1d(data.unsqueeze(1), k, padding=pad).squeeze(1)
        x = _reflect_pad(data.unsqueeze(1), pad, dim=2)
        return torch.nn.functional.conv1d(x, k, padding=0).squeeze(1)

    if n_spatial == 2:
        k2d = (kernel[:, None] * kernel[None, :]).reshape(1, 1, -1, len(kernel))
        if boundary == "absorbing":
            return torch.nn.functional.conv2d(data.unsqueeze(1), k2d, padding=pad).squeeze(1)
        x = _reflect_pad(_reflect_pad(data.unsqueeze(1), pad, dim=2), pad, dim=3)
        return torch.nn.functional.conv2d(x, k2d, padding=0).squeeze(1)

    raise NotImplementedError(f"PSF not implemented for {n_spatial}D spatial data in torch")


def _generate_bold_noise(shape: tuple[int, ...], noise: Noise, amplitude: float) -> np.ndarray:
    rng = np.random.default_rng(noise.seed)

    if noise.type == "white":
        return rng.normal(0.0, amplitude, size=shape).astype(np.float64)

    if noise.type == "uniform":
        return rng.uniform(-amplitude, amplitude, size=shape).astype(np.float64)

    if noise.type == "pink":
        T = shape[0]
        spatial_shape = shape[1:]
        n_spatial = int(np.prod(spatial_shape)) if spatial_shape else 1

        freqs = np.fft.rfftfreq(T)
        freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
        shaping = 1.0 / np.sqrt(freqs)

        out = np.empty(shape, dtype=np.float64)
        flat = out.reshape(T, n_spatial)
        for k in range(n_spatial):
            x = rng.standard_normal(T)
            X = np.fft.rfft(x)
            X *= shaping
            y = np.fft.irfft(X, n=T)
            y = (y - y.mean()) / (y.std() + 1e-12)
            flat[:, k] = amplitude * y
        return out

    raise ValueError(f"Unknown noise type: {noise.type}")


def _is_torch(x: object) -> bool:
    return isinstance(x, torch.Tensor)


def _get_torch_ref(xs: list[Timecourse]) -> torch.Tensor | None:
    for x in xs:
        if isinstance(x, torch.Tensor):
            return x
    return None


def _finite(x: Timecourse, max_abs: float) -> Timecourse:
    if isinstance(x, torch.Tensor):
        return torch.nan_to_num(x, nan=0.0, posinf=max_abs, neginf=-max_abs)
    return np.nan_to_num(x, nan=0.0, posinf=max_abs, neginf=-max_abs)


def _clamp_pos(x: Timecourse, eps: float, max_abs: float) -> Timecourse:
    x = _finite(x, max_abs)
    if isinstance(x, torch.Tensor):
        return torch.clamp(x, min=eps)
    return np.maximum(x, eps)


def _clamp_any(x: Timecourse, max_abs: float) -> Timecourse:
    x = _finite(x, max_abs)
    if isinstance(x, torch.Tensor):
        return torch.clamp(x, min=-max_abs, max=max_abs)
    return np.clip(x, -max_abs, max_abs)


def sanitize_state(
    values: Mapping[str, Timecourse | None],
    *,
    eps: float = 1e-6,
    max_abs: float = 1e3,
    positive: tuple[str, ...] = ("f", "v", "q"),
    signed: tuple[str, ...] = ("x", "s", "v*", "q*"),
) -> dict[str, Timecourse | None]:
    out: dict[str, Timecourse | None] = dict(values)

    for k in positive:
        v = out.get(k, None)
        if v is not None:
            out[k] = _clamp_pos(v, eps, max_abs)

    for k in signed:
        v = out.get(k, None)
        if v is not None:
            out[k] = _clamp_any(v, max_abs)

    return out


def balloon_derivatives(layer: CortexLayer, c: HaemodynamicConstants, order: str) -> dict[str, Timecourse]:
    clean = sanitize_state(layer.state.as_dict())
    x = clean["x"]  # type: ignore[assignment]
    s = clean["s"]  # type: ignore[assignment]
    f = clean["f"]  # type: ignore[assignment]
    v = clean["v"]  # type: ignore[assignment]
    q = clean["q"]  # type: ignore[assignment]

    ds = x - c.kappa * s - c.gamma * (f - 1.0)  # type: ignore[operator]
    df = s  # type: ignore[assignment]

    if order == "exact":
        outflow = v ** (1.0 / c.alpha)  # type: ignore[operator]
        extraction = (1.0 - (1.0 - c.E0) ** (1.0 / f)) / c.E0  # type: ignore[operator]

        dv = (f - outflow) / layer.tau  # type: ignore[operator]
        dq = (f * extraction - outflow * (q / v)) / layer.tau  # type: ignore[operator]

    elif order == "linear":
        v -= 1
        f -= 1
        q -= 1
        beta = (1 - c.E0) * np.log(1 - c.E0) / c.E0 
        dv = (f - v / c.alpha) / layer.tau  
        dq = ( (1 + beta) * f - q - (1/c.alpha - 1) * v ) / layer.tau 
        v += 1
        f += 1
        q += 1

    elif order == "quadratic":
        v -= 1
        f -= 1
        q -= 1
        beta = (1 - c.E0) * np.log(1 - c.E0) / c.E0
        gamma = beta * np.log(1 - c.E0) / 2
        dv = (f - v / c.alpha - (1 - c.alpha) / (2 * c.alpha**2) * v**2) / layer.tau
        dq = ((1 + beta) * f - q - (1/c.alpha - 1) * v
            - gamma * f**2
            - (1/c.alpha - 1) * v * q
            - (1/2) * (1/c.alpha - 1) * (1/c.alpha - 2) * v**2) / layer.tau        
        v += 1
        f += 1
        q += 1

    else:
        raise ValueError(f"Unknown order: '{order}'. Implemented 'exact', 'linear', or 'quadratic'.")


    if layer.drain_from is not None and layer.lambda_d != 0.0:
        delayed = sanitize_state(layer.drain_from.state.as_dict())
        v_star = delayed.get("v*", None)
        q_star = delayed.get("q*", None)
        if v_star is None or q_star is None:
            raise ValueError("drain_from exists but lower layer v_star/q_star are None.")
        dv = dv + (layer.lambda_d * v_star) / layer.tau  # type: ignore[operator]
        dq = dq + (layer.lambda_d * q_star) / layer.tau  # type: ignore[operator]

    return {"ds_dt": ds, "df_dt": df, "dv_dt": dv, "dq_dt": dq}  # type: ignore[return-value]

def delay_filter_derivatives(layer: CortexLayer, *, tau_d: float) -> dict[str, Timecourse]:
    if layer.state.v_star is None or layer.state.q_star is None:
        raise ValueError("Delayed states are None. Allocate v_star and q_star before calling.")

    clean = sanitize_state(
        {"v": layer.state.v, "q": layer.state.q, "v*": layer.state.v_star, "q*": layer.state.q_star}
    )
    v = clean["v"]  # type: ignore[assignment]
    q = clean["q"]  # type: ignore[assignment]
    v_star = clean["v*"]  # type: ignore[assignment]
    q_star = clean["q*"]  # type: ignore[assignment]

    dv_star = (-v_star + (v - 1.0)) / tau_d  # type: ignore[operator]
    dq_star = (-q_star + (q - 1.0)) / tau_d  # type: ignore[operator]

    # Cleaner derivative names (avoid '*' in keys)
    return {"dv_star_dt": dv_star, "dq_star_dt": dq_star}

def get_inversion_derivatives(
    layer: CortexLayer, c: HaemodynamicConstants, tau_d: float, order: str
) -> dict[str, Timecourse]:
    delayed_derivs = delay_filter_derivatives(layer, tau_d=tau_d)
    balloon_derivs = balloon_derivatives(layer, c, order)
    return {**balloon_derivs, **delayed_derivs}











def rk4_step(
    y: dict[str, Timecourse],
    dy_fn,
    dt: float,
    *,
    state_keys: tuple[str, ...],
    deriv_keys: dict[str, str],
) -> dict[str, Timecourse]:
    k1 = dy_fn(y)

    # fail fast if derivative dict doesn't match mapping
    for k in state_keys:
        dk = deriv_keys[k]
        if dk not in k1:
            raise KeyError(
                f"Missing derivative '{dk}' for state '{k}'. Available: {sorted(k1.keys())}"
            )

    y2 = dict(y)
    for k in state_keys:
        y2[k] = y[k] + 0.5 * dt * k1[deriv_keys[k]]  # type: ignore[operator]
    k2 = dy_fn(y2)

    y3 = dict(y)
    for k in state_keys:
        y3[k] = y[k] + 0.5 * dt * k2[deriv_keys[k]]  # type: ignore[operator]
    k3 = dy_fn(y3)

    y4 = dict(y)
    for k in state_keys:
        y4[k] = y[k] + dt * k3[deriv_keys[k]]  # type: ignore[operator]
    k4 = dy_fn(y4)

    out = dict(y)
    for k in state_keys:
        out[k] = y[k] + (dt / 6.0) * (  # type: ignore[operator]
            k1[deriv_keys[k]]
            + 2.0 * k2[deriv_keys[k]]
            + 2.0 * k3[deriv_keys[k]]
            + k4[deriv_keys[k]]
        )
    return out


def simulate_cortex(
    layers: list[CortexLayer],
    constants: HaemodynamicConstants,
    x_inputs: list[Timecourse],
    *,
    dt: float,
    tau_d: float, 
    order: str,
) -> dict[int, dict[str, Timecourse]]:
    if len(layers) != len(x_inputs):
        raise ValueError(f"len(layers)={len(layers)} must match len(x_inputs)={len(x_inputs)}")

    T = int(x_inputs[0].shape[0])
    D = tuple(x_inputs[0].shape[1:])

    for i, xi in enumerate(x_inputs):
        if tuple(xi.shape) != (T, *D):
            raise ValueError(f"x_inputs[{i}] shape {tuple(xi.shape)} != {(T, *D)}")

    use_torch = any(_is_torch(xi) for xi in x_inputs)
    ref = _get_torch_ref(x_inputs)

    def zeros(T_: int, spatial_shape: tuple[int, ...]) -> Timecourse:
        shape = (T_,) + spatial_shape
        if use_torch:
            assert ref is not None
            return torch.zeros(shape, dtype=ref.dtype, device=ref.device)
        return np.zeros(shape, dtype=np.asarray(x_inputs[0]).dtype)

    out: dict[int, dict[str, Timecourse]] = {}
    for layer in layers:
        out[layer.depth] = {
            "x": zeros(T, D),
            "s": zeros(T, D),
            "f": zeros(T, D),
            "v": zeros(T, D),
            "q": zeros(T, D),
        }
        if layer.state.v_star is not None and layer.state.q_star is not None:
            out[layer.depth]["v*"] = zeros(T, D)
            out[layer.depth]["q*"] = zeros(T, D)

    # time loop
    for t in range(T):
        # forcing
        for i, layer in enumerate(layers):
            layer.state.x = x_inputs[i][t]

        # 1) delay update
        for layer in layers:
            if layer.state.v_star is None or layer.state.q_star is None:
                continue

            y_delay = {
                "v": layer.state.v,
                "q": layer.state.q,
                "v*": layer.state.v_star,
                "q*": layer.state.q_star,
            }

            def delay_dy_fn(ycand: dict[str, Timecourse], layer=layer) -> dict[str, Timecourse]:
                layer.state.v = ycand["v"]
                layer.state.q = ycand["q"]
                layer.state.v_star = ycand["v*"]
                layer.state.q_star = ycand["q*"]
                return delay_filter_derivatives(layer, tau_d=tau_d)

            y_delay_next = rk4_step(
                y_delay,
                delay_dy_fn,
                dt,
                state_keys=("v*", "q*"),
                deriv_keys={"v*": "dv_star_dt", "q*": "dq_star_dt"},
            )
            layer.state.v_star = y_delay_next["v*"]
            layer.state.q_star = y_delay_next["q*"]

        # 2) balloon update
        for layer in layers:
            y = {
                "x": layer.state.x,
                "s": layer.state.s,
                "f": layer.state.f,
                "v": layer.state.v,
                "q": layer.state.q,
            }

            def balloon_dy_fn(ycand: dict[str, Timecourse], layer=layer) -> dict[str, Timecourse]:
                layer.state.x = ycand["x"]
                layer.state.s = ycand["s"]
                layer.state.f = ycand["f"]
                layer.state.v = ycand["v"]
                layer.state.q = ycand["q"]
                return balloon_derivatives(layer, constants, order)

            y_next = rk4_step(
                y,
                balloon_dy_fn,
                dt,
                state_keys=("s", "f", "v", "q"),
                deriv_keys={"s": "ds_dt", "f": "df_dt", "v": "dv_dt", "q": "dq_dt"},
            )

            layer.state.s = y_next["s"]
            layer.state.f = y_next["f"]
            layer.state.v = y_next["v"]
            layer.state.q = y_next["q"]

        # 3) record
        for layer in layers:
            d = out[layer.depth]
            d["x"][t] = layer.state.x  # type: ignore[index]
            d["s"][t] = layer.state.s  # type: ignore[index]
            d["f"][t] = layer.state.f  # type: ignore[index]
            d["v"][t] = layer.state.v  # type: ignore[index]
            d["q"][t] = layer.state.q  # type: ignore[index]
            if "v*" in d and layer.state.v_star is not None:
                d["v*"][t] = layer.state.v_star  # type: ignore[index]
            if "q*" in d and layer.state.q_star is not None:
                d["q*"][t] = layer.state.q_star  # type: ignore[index]

    return out


def get_bold_from_state(
    state: dict[str, Timecourse],
    acq: AcquisitionConstants,
    c: HaemodynamicConstants,
    *,
    layer_depth: int | None = None,
    params: BoldPostProcessingConfig | None = None,
) -> Timecourse:
    k1 = acq.k1
    k2 = acq.k2
    k3 = acq.k3
    V0 = c.V0

    q = state["q"]
    v = state["v"]

    bold = V0 * (k1 * (1 - q) + k2 * (1 - q / v) + k3 * (1 - v))

    if params is None:
        return bold

    # PSF convolution (spatial blurring)
    if params.layer_psf is not None and layer_depth is not None:
        psf = params.layer_psf.get(layer_depth)
        if psf is not None:
            bold = psf.apply(bold)

    # Noise amplitude priority: snr_db > noise_models > noise_amplitude
    amplitude = params.noise_amplitude
    if params.noise_models is not None:
        amplitude = sum(nm.noise_std * scale for nm, scale in params.noise_models)
    if params.snr_db is not None:
        if params.snr_db == np.inf:
            amplitude = 0.0
        else:
            if isinstance(bold, torch.Tensor):
                p_signal = float(torch.mean(bold**2).item())
            else:
                p_signal = float(np.mean(bold**2))
            snr_linear = 10.0 ** (float(params.snr_db) / 10.0)
            amplitude = float(np.sqrt(p_signal / snr_linear))

    if params.noise is not None and amplitude > 0.0:
        noise_arr = _generate_bold_noise(bold.shape, params.noise, amplitude)
        if isinstance(bold, torch.Tensor):
            bold = bold + torch.as_tensor(noise_arr, dtype=bold.dtype, device=bold.device)
        else:
            bold = bold + noise_arr

    return bold



