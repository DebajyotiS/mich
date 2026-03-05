from dataclasses import dataclass

import numpy as np
from scipy.ndimage import convolve

from .signals import Noise, Pulse


@dataclass(frozen=True, slots=True)
class NeuralSimulatorParams:
    num_layers: int
    grid_size: tuple[int, int]
    dt: float = 0.01
    dx: float = 1.0
    diffusion_coefficient_inter: float = 1.0
    diffusion_coefficient_intra: float = 0.1

    # stability / integration controls
    safety: float = 0.9  # CFL safety factor
    max_substeps: int = 128  # cap work per step
    noise_as_sde: bool = True  # scale noise with sqrt(dt)


class LayeredDiffusionSimulator:
    """Simulates pulse signals with multiple peaks diffusing
    within and through a stack of layers.
    """

    def __init__(self, params: NeuralSimulatorParams):
        """

        Args:
            params (NeuralSimulatorParams): Simulation parameters.
        """
        self.params = params
        self.n_layers = params.num_layers
        self.grid_size = params.grid_size
        self.dt = params.dt
        self.dx = params.dx
        self.diff_inter = params.diffusion_coefficient_inter
        self.diff_intra = params.diffusion_coefficient_intra
        self.grid = np.zeros((params.num_layers, params.grid_size[0], params.grid_size[1]))
        self.safety = params.safety
        self.max_substeps = params.max_substeps

    def generate_pulse(self, pulses: list[Pulse]) -> list[tuple[np.ndarray, np.ndarray]]:
        signals = []
        for pulse in pulses:
            t, signal = pulse.generate()
            signals.append((t, signal))
        return signals

    def simulate(
        self,
        sources: dict | list[dict],
        steps: int,
        snr_db: float,
        noise: Noise,
    ):
        if steps is None or steps <= 0:
            raise ValueError(f"`steps` must be a positive int, got {steps}")

        # Normalize sources to list[dict]
        if isinstance(sources, dict):
            sources = [sources]

        # Validate source dicts + signals
        for k, src in enumerate(sources):
            if "layer" not in src or "position" not in src or "signal" not in src:
                raise ValueError("Each source must have 'layer', 'position', and 'signal' keys.")
            layer = int(src["layer"])
            if not (0 <= layer < self.n_layers):
                raise ValueError(
                    f"source[{k}] has invalid layer={layer} (n_layers={self.n_layers})"
                )
            pos = tuple(src["position"])
            if len(pos) != 2:
                raise ValueError(f"source[{k}] position must be (i,j), got {pos}")
            i, j = int(pos[0]), int(pos[1])
            if not (0 <= i < self.grid_size[0] and 0 <= j < self.grid_size[1]):
                raise ValueError(
                    f"source[{k}] position out of bounds: {pos} for grid {self.grid_size}"
                )

            sig = np.asarray(src["signal"], dtype=np.float64)
            if sig.ndim != 1:
                raise ValueError(f"source[{k}] signal must be 1D, got shape {sig.shape}")
            if not np.isfinite(sig).all():
                bad = np.where(~np.isfinite(sig))[0][:10]
                raise ValueError(f"source[{k}] signal contains non-finite values at indices {bad}")
            src["signal"] = sig

        # Param sanity
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError(f"Bad dt={self.dt}")
        if not np.isfinite(self.dx) or self.dx <= 0:
            raise ValueError(f"Bad dx={self.dx}")
        if self.diff_intra < 0 or self.diff_inter < 0:
            raise ValueError("Diffusion coefficients must be non-negative.")

        # Compute noise amplitude from SNR (based on max source signal power)
        max_signal_power = 0.0
        for src in sources:
            sig = src["signal"]
            p = float(np.mean(sig * sig))
            if p > max_signal_power:
                max_signal_power = p

        if snr_db == np.inf:
            Pn_total = 0.0
        else:
            snr_linear = 10.0 ** (float(snr_db) / 10.0)
            Pn_total = max_signal_power / snr_linear

        domain = getattr(noise, "domain", "spatial")  # backward compatible
        tf_default = getattr(noise, "temporal_fraction", 0.5)

        if domain == "temporal":
            tf = 1.0
        elif domain == "spatial":
            tf = 0.0
        elif domain == "both":
            tf = float(tf_default)
        else:
            raise ValueError(f"Unknown noise.domain: {domain}")

        tf = float(np.clip(tf, 0.0, 1.0))
        sf = 1.0 - tf

        noise_amp_temporal = np.sqrt(Pn_total * tf)
        noise_amp_spatial = np.sqrt(Pn_total * sf)

        # Use float64 state/history to reduce overflow risk
        self.grid = np.zeros(
            (self.n_layers, self.grid_size[0], self.grid_size[1]), dtype=np.float64
        )
        history = np.zeros(
            (steps, self.n_layers, self.grid_size[0], self.grid_size[1]), dtype=np.float64
        )

        # Discrete Laplacian kernel (already includes 1/dx^2)
        laplacian_kernel = np.array(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        ) / (self.dx**2)

        # Conservative combined bound for 2D intra-layer + 1D inter-layer coupling.
        den = 4.0 * self.diff_intra + 2.0 * self.diff_inter
        dt_max = (self.dx**2) / den if den > 0 else np.inf

        safety = self.safety
        max_substeps = self.max_substeps

        if np.isfinite(dt_max) and self.dt > safety * dt_max:
            required = int(np.ceil(self.dt / (safety * dt_max)))
            if required > max_substeps:
                raise ValueError(
                    f"Unstable explicit diffusion for dt={self.dt}. "
                    f"Need ~{required} substeps (dt_max≈{dt_max:.3g}), "
                    f"but max_substeps={max_substeps}. Reduce dt, increase dx, or reduce diffusion."
                )
            n_sub = required
        else:
            n_sub = 1

        dt_sub = self.dt / n_sub

        # Noise scaling: treat injected noise as SDE-like => sqrt(dt)
        noise_scale = np.sqrt(self.dt)

        # -----------------------------
        # NEW: optional temporal noise
        # -----------------------------
        n_sources = len(sources)
        temporal_noise = None

        # Backwards compatible:
        # - If Noise has no attribute "domain", behave like old code (spatial only).
        # - If domain is "temporal"/"both", require generate_temporal.
        domain = getattr(noise, "domain", "spatial")

        if domain in ("temporal", "both") and noise_amp_temporal > 0.0:
            gen_temporal = getattr(noise, "generate_temporal", None)
            if gen_temporal is None:
                raise AttributeError(
                    "noise.domain requests temporal noise, but Noise has no generate_temporal(...) method."
                )
            temporal_noise = np.asarray(
                gen_temporal(noise_amp_temporal, n_sources, steps, self.dt),
                dtype=np.float64,
            )
            if temporal_noise.shape != (n_sources, steps):
                raise ValueError(
                    f"Temporal noise has shape {temporal_noise.shape}, expected {(n_sources, steps)}"
                )
            if not np.isfinite(temporal_noise).all():
                raise ValueError("Noise.generate_temporal returned non-finite values.")

        # Main loop
        for step in range(steps):
            # -----------------------------
            # Spatial noise (2D per layer)
            # -----------------------------
            if domain in ("spatial", "both") and noise_amp_spatial > 0.0:
                noise_grid = noise.generate(noise_amp_spatial, self.n_layers, self.grid_size)
                noise_grid = np.asarray(noise_grid, dtype=np.float64)
                if noise_grid.shape != self.grid.shape:
                    raise ValueError(
                        f"Noise grid has shape {noise_grid.shape}, expected {self.grid.shape}"
                    )
                if not np.isfinite(noise_grid).all():
                    raise ValueError("Noise.generate returned non-finite values.")
                self.grid += noise_grid * noise_scale

            # -----------------------------
            # Inject sources (+ temporal noise)
            # -----------------------------
            for s_idx, src in enumerate(sources):
                layer = int(src["layer"])
                i, j = map(int, src["position"])
                sig = src["signal"]

                if step < sig.shape[0]:
                    inj = float(sig[step])
                    if temporal_noise is not None:
                        inj += float(temporal_noise[s_idx, step])
                    self.grid[layer, i, j] += inj * self.dt

            if not np.isfinite(self.grid).all():
                where = np.argwhere(~np.isfinite(self.grid))[0]
                layer_index, i, j = map(int, where)
                raise FloatingPointError(
                    f"Non-finite in grid BEFORE diffusion at step={step}, "
                    f"layer={layer_index}, pos=({i},{j}), val={self.grid[layer_index, i, j]}"
                )

            # Diffusion substeps
            for _ in range(n_sub):
                new_grid = self.grid.copy()

                for layer_index in range(self.n_layers):
                    lap = convolve(self.grid[layer_index], laplacian_kernel, mode="constant", cval=0.0)
                    new_grid[layer_index] += (self.diff_intra * lap) * dt_sub

                    # inter-layer coupling (second difference across layers)
                    if layer_index > 0:
                        flux_below = (
                            self.diff_inter * (self.grid[layer_index - 1] - self.grid[layer_index]) / (self.dx**2)
                        )
                        new_grid[layer_index] += flux_below * dt_sub
                    if layer_index < self.n_layers - 1:
                        flux_above = (
                            self.diff_inter * (self.grid[layer_index + 1] - self.grid[layer_index]) / (self.dx**2)
                        )
                        new_grid[layer_index] += flux_above * dt_sub

                self.grid = new_grid

                if not np.isfinite(self.grid).all():
                    where = np.argwhere(~np.isfinite(self.grid))[0]
                    layer_index, i, j = map(int, where)
                    raise FloatingPointError(
                        f"Non-finite in grid AFTER diffusion substep at outer step={step}, "
                        f"layer={layer_index}, pos=({i},{j}), val={self.grid[layer_index, i, j]}"
                    )

            history[step] = self.grid.copy()

        return history
