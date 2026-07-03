from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from mich import CONFIG_DIR
from mich.data.balloon import (
    AcquisitionConstants,
    BoldPostProcessingConfig,
    CortexLayer,
    HaemodynamicConstants,
    HaemodynamicState,
    NoiseModel,
    PointSpreadFunction,
    get_bold_from_state,
    simulate_cortex,
)
from mich.data.neuronal import LayeredDiffusionSimulator, NeuralSimulatorParams
from mich.data.signals import Noise, Pulse, Sources

# Maps pulse_type -> ordered extra parameter names (following amplitude and onset).
_PULSE_EXTRA_PARAMS: dict[str, list[str]] = {
    "rect": ["width"],
    "exp_decay": ["decay_rate"],
}


def _draw_pulse_signal(
    rng: np.random.Generator,
    pulse_type: str,
    max_pulses: int,
    amp_range: tuple[float, float],
    width_range: tuple[float, float],
    isi_min: int,
    time_duration: int,
    dt: float,
    baseline_mode: str,
) -> tuple[list[list[float]], np.ndarray]:
    """Draw a random pulse train for one source and generate its signal waveform."""
    num_pulses = int(rng.integers(1, max_pulses + 1))
    amplitudes = rng.uniform(*amp_range, size=num_pulses)
    if pulse_type == "rect":
        widths = rng.uniform(*width_range, size=num_pulses)

    init_time_gap: int = 10
    onset_times: list[float] = []
    valid_widths: list[float] = []

    for i in range(num_pulses):
        time = float(rng.uniform(init_time_gap, init_time_gap + isi_min))
        pulse_width = float(widths[i]) if pulse_type == "rect" else float(rng.uniform(0.1, 1.0))
        pulse_end_time = time + pulse_width

        if pulse_end_time > time_duration:
            break

        onset_times.append(time)
        valid_widths.append(pulse_width)

        init_time_gap = pulse_end_time + isi_min
        if init_time_gap > time_duration:
            break

    pulse_list = [
        [float(amplitudes[k]), onset_times[k], valid_widths[k]] for k in range(len(onset_times))
    ]

    pulse = Pulse(
        pulse_type=pulse_type,
        peaks=pulse_list,
        duration=time_duration,
        dt=dt,
        baseline=baseline_mode,
        rng=rng,
    )
    return pulse_list, pulse.generate()[1]


def _derive_acquisition(cfg: dict) -> AcquisitionConstants:
    ac = cfg["acquisition"]
    hc = cfg["haemodynamic"]
    k1 = 4.3 * ac["f0"] * hc["E0"] * ac["TE"]
    k2 = ac["eps"] * ac["r0"] * hc["E0"] * ac["TE"]
    k3 = 1.0 - ac["eps"]
    return AcquisitionConstants(k1=k1, k2=k2, k3=k3)


def _derive_haemo(cfg: dict) -> HaemodynamicConstants:
    hc = cfg["haemodynamic"]
    return HaemodynamicConstants(
        kappa=hc["kappa"],
        gamma=hc["gamma"],
        alpha=hc["alpha"],
        E0=hc["E0"],
        V0=hc["V0"],
    )


def _derive_psf_fwhm(bold_cfg: dict, num_layers: int) -> list[float]:
    """Interpolate per-layer PSF FWHM (grid units) from physical parameters.

    bold.psf_fwhm_mm is [deepest, most_superficial] FWHM in mm; intermediate
    layers are linearly interpolated by normalized cortical depth. Dividing by
    bold.voxel_mm (the fixed physical pixel size, independent of grid_size)
    converts to the grid units PointSpreadFunction expects.
    """
    voxel_mm: float = bold_cfg["voxel_mm"]
    deep_mm, sup_mm = bold_cfg["psf_fwhm_mm"]
    if num_layers == 1:
        fwhm_mm = [deep_mm]
    else:
        fwhm_mm = [deep_mm + (i / (num_layers - 1)) * (sup_mm - deep_mm) for i in range(num_layers)]
    return [f / voxel_mm for f in fwhm_mm]


def _run_neural(
    cfg: dict,
    rng: np.random.Generator,
    sim_params: NeuralSimulatorParams,
    steps: int,
    seed: int | None,
) -> tuple[list, list, list[int], list[tuple], int]:
    """Generate stimuli, place sources based on structural config boundaries, and run neural diffusion."""
    sc = cfg["simulation"]
    num_layers: int = sc["num_layers"]
    grid_size = tuple(sc["grid_size"])
    dt: float = sc["dt"]
    time_duration: int = sc["time_duration"]
    max_pulses: int = sc["max_pulses"]
    isi_min: int = sc.get("isi_min", 20)
    pulse_type: str = sc.get("pulse_type", "rect")

    # Extract clean boundaries with safe fallbacks mimicking legacy behavior [1 layer, 1 source]
    placement_cfg = sc.get("source_placement", {})
    src_per_layer_range = placement_cfg.get("sources_per_layer", [1, 1])
    shared_position: bool = placement_cfg.get("shared_position", False)
    shared_pulse: bool = placement_cfg.get("shared_pulse", False)
    if shared_position and tuple(src_per_layer_range) != (1, 1):
        raise ValueError(
            "source_placement.shared_position requires sources_per_layer == [1, 1] "
            f"(got {src_per_layer_range}); a shared column position only makes sense "
            "with exactly one source per active layer"
        )
    if shared_pulse and not shared_position:
        raise ValueError(
            "source_placement.shared_pulse requires source_placement.shared_position "
            "to also be true -- a shared pulse only makes sense when layers also share a position"
        )

    amp_range: tuple[float, float] = tuple(sc.get("amp_range", (0.1, 1.0)))
    width_range: tuple[float, float] = tuple(sc.get("width_range", (2.0, 10.0)))
    baseline_mode: str = sc.get("baseline_mode", "random")

    if pulse_type not in ["rect", "exp_decay"]:
        raise ValueError(f"Unknown pulse_type {pulse_type!r}. Must be 'rect' or 'exp_decay'.")

    # Dynamically evaluate active layers based on constraints
    min_active_layers: int = sc.get("min_active_layers", 1)
    max_active_layers: int = sc.get("max_active_layers", 1)
    lo = max(1, min(min_active_layers, num_layers))
    hi = min(max_active_layers, num_layers)
    if lo > hi:
        raise ValueError(
            f"min_active_layers ({min_active_layers}) must be <= max_active_layers ({max_active_layers})"
        )
    num_active_layers = int(rng.integers(lo, hi + 1))
    chosen_layers = rng.choice(num_layers, size=num_active_layers, replace=False).tolist()

    # When shared_position is set, every active layer's (single) source sits at the
    # same (x, y) -- a cortical column -- instead of each layer drawing its own position.
    shared_pos: tuple[int, int] | None = None
    if shared_position:
        shared_pos = tuple(rng.integers(0, grid_size[0], size=2).tolist())

    # When shared_pulse is set, every active layer's source also reuses the exact same
    # pulse train/waveform, drawn once here, instead of each layer drawing its own.
    shared_pulse_data: tuple[list[list[float]], np.ndarray] | None = None
    if shared_pulse:
        shared_pulse_data = _draw_pulse_signal(
            rng,
            pulse_type,
            max_pulses,
            amp_range,
            width_range,
            isi_min,
            time_duration,
            dt,
            baseline_mode,
        )

    sources = Sources()
    all_pulse_lists = []
    active_layers = []
    active_positions = []
    total_pulses_count = 0

    for layer in chosen_layers:
        # Evaluate how many discrete coordinates to spawn in this specific layer plane
        num_positions = int(rng.integers(src_per_layer_range[0], src_per_layer_range[1] + 1))

        for _ in range(num_positions):
            pos = (
                shared_pos
                if shared_position
                else tuple(rng.integers(0, grid_size[0], size=2).tolist())
            )

            if shared_pulse_data is not None:
                pulse_list, signal = shared_pulse_data
            else:
                pulse_list, signal = _draw_pulse_signal(
                    rng,
                    pulse_type,
                    max_pulses,
                    amp_range,
                    width_range,
                    isi_min,
                    time_duration,
                    dt,
                    baseline_mode,
                )

            sources.add_source(layer=layer, position=pos, signal=signal)

            all_pulse_lists.append(pulse_list)
            active_layers.append(layer)
            active_positions.append(pos)
            total_pulses_count += len(pulse_list)

    # Run execution loop through the coupled simulator framework
    neural_noise = Noise(type="pink", seed=seed, domain="both")
    simulator = LayeredDiffusionSimulator(sim_params)
    history = simulator.simulate(
        sources=sources.get_sources(),
        noise=neural_noise,
        steps=steps,
        snr_db=float(sc["neural_SNR"]),
    )

    x_inputs = [history[:, i, ...] for i in range(num_layers)]
    return x_inputs, all_pulse_lists, active_layers, active_positions, total_pulses_count


def _run_haemo_and_bold(
    cfg: dict,
    x_inputs: list,
    haemo: HaemodynamicConstants,
    acq: AcquisitionConstants,
    seed: int | None,
) -> tuple[dict, dict]:
    """Run haemodynamic simulation and compute BOLD signals for all layers.

    Returns (out, bold_signals) where out is the per-layer haemodynamic state dict
    and bold_signals maps layer index to BOLD array.
    """
    sc = cfg["simulation"]
    num_layers: int = sc["num_layers"]
    layers_cfg: list[dict] = sc["layers"]
    dt: float = sc["dt"]
    order: str = sc["order"]
    haemo_dt: float = sc.get("haemo_dt", dt)

    haemo_ratio = round(dt / haemo_dt)
    if haemo_ratio < 1:
        raise ValueError(f"haemo_dt ({haemo_dt}) must be <= dt ({dt})")

    if haemo_ratio > 1:
        x_inputs_haemo = [np.repeat(xi, haemo_ratio, axis=0) for xi in x_inputs]
    else:
        x_inputs_haemo = x_inputs

    cortex_layers: list[CortexLayer] = []
    for i, lc in enumerate(layers_cfg):
        x_i = x_inputs_haemo[i]
        state = HaemodynamicState(
            x=x_i[0],
            s=np.zeros_like(x_i[0]),
            f=np.ones_like(x_i[0]),
            v=np.ones_like(x_i[0]),
            q=np.ones_like(x_i[0]),
            v_star=np.zeros_like(x_i[0]),
            q_star=np.zeros_like(x_i[0]),
        )
        cortex_layers.append(
            CortexLayer(
                depth=i,
                tau=lc["tau"],
                state=state,
                lambda_d=lc.get("lambda_d", 0.0),
                drain_from=cortex_layers[i - 1] if i > 0 else None,
            )
        )

    out = simulate_cortex(
        cortex_layers,
        haemo,
        x_inputs=x_inputs_haemo,  # type: ignore
        dt=haemo_dt,
        tau_d=sc["tau_d"],
        order=order,
    )

    if haemo_ratio > 1:
        out = {
            li: {k: (v[::haemo_ratio] if v is not None else None) for k, v in ld.items()}
            for li, ld in out.items()
        }

    bold_cfg = cfg.get("bold", {})
    psf_fwhm: list = bold_cfg["psf_fwhm"]

    nm_cfg = bold_cfg.get("noise_model", {})
    noise_model = NoiseModel.preset(
        nm_cfg.get("field", "7T"),
        V=nm_cfg.get("V", 8.0),
        TR=nm_cfg.get("TR", 2.0),
    )
    noise_scales = bold_cfg.get("noise_scales", [1.0, 1.23, 1.16])

    bold_params = BoldPostProcessingConfig(
        layer_psf={i: PointSpreadFunction(fwhm=psf_fwhm[i]) for i in range(num_layers)},
        noise=Noise(type="white", seed=seed, domain="both"),
        snr_db=float(sc["BOLD_SNR"]),
        noise_models=[(noise_model, s) for s in noise_scales],
    )

    bold_signals = {
        i: get_bold_from_state(
            out[i],  # type: ignore
            acq,
            haemo,
            layer_depth=i,
            params=bold_params,
        )
        for i in range(num_layers)
    }

    return out, bold_signals


def run_simulation(cfg: dict, seed: int | None = None) -> dict:
    """Run one complete simulation and return a results dict."""
    rng = np.random.default_rng(seed)
    sc = cfg["simulation"]

    num_layers: int = sc["num_layers"]
    grid_size = tuple(sc["grid_size"])
    dt: float = sc["dt"]
    time_duration: int = sc["time_duration"]
    steps: int = int(time_duration / dt)

    layers_cfg: list[dict] = sc["layers"]
    bold_cfg = cfg.get("bold", {})
    psf_fwhm = bold_cfg.get("psf_fwhm", [])
    noise_scales = bold_cfg.get("noise_scales", [])
    _bad = {
        "simulation.layers": len(layers_cfg),
        "bold.psf_fwhm": len(psf_fwhm),
        "bold.noise_scales": len(noise_scales),
    }
    if any(v != num_layers for v in _bad.values()):
        detail = ", ".join(f"{k}={v}" for k, v in _bad.items())
        raise ValueError(
            f"All layer-indexed lists must have length num_layers={num_layers}: {detail}"
        )

    acq = _derive_acquisition(cfg)
    haemo = _derive_haemo(cfg)
    sim_params = NeuralSimulatorParams(
        num_layers=num_layers,
        grid_size=grid_size,
        diffusion_coefficient_intra=sc["diffusion_coefficient_intra"],
        diffusion_coefficient_inter=sc["diffusion_coefficient_inter"],
        dt=dt,
        decay_rate=sc.get("decay_rate", 0.5),
    )

    x_inputs, pulse_list, source_layer, source_pos, num_pulses = _run_neural(
        cfg, rng, sim_params, steps, seed
    )
    out, bold_signals = _run_haemo_and_bold(cfg, x_inputs, haemo, acq, seed)

    results: dict = {"layers": {}, "meta": {}}
    for i in range(num_layers):
        results["layers"][i] = {
            "s": out[i]["s"],
            "f": out[i]["f"],
            "v": out[i]["v"],
            "q": out[i]["q"],
            "v_star": out[i].get("v*"),
            "q_star": out[i].get("q*"),
            "x": out[i]["x"],
            "bold": bold_signals[i],
        }

    results["meta"] = {
        "pulses": pulse_list,
        "source_layer": source_layer,
        "source_position": list(source_pos),
        "seed": seed,
        "num_pulses": num_pulses,
    }
    return results


# Keys needed for training -- stored at full T resolution
TRAIN_KEYS = ("x", "bold")
# Intermediate latent states -- stored at reduced resolution for inspection only
LATENT_KEYS = ("s", "f", "v", "q", "v_star", "q_star")


def init_h5(
    h5f: h5py.File,
    cfg: dict,
    num_sims: int,
    latent_downsample: int = 10,
) -> None:

    if not isinstance(latent_downsample, int) or latent_downsample < 1:
        raise ValueError(f"latent_downsample must be a positive int, got {latent_downsample!r}")

    sc = cfg["simulation"]
    num_layers = sc["num_layers"]
    grid_size = tuple(sc["grid_size"])
    T = int(sc["time_duration"] / sc["dt"])
    T_lat = T // latent_downsample

    max_pulses = sc["max_pulses"]
    placement_cfg = sc["source_placement"]
    max_sources = num_layers * placement_cfg["sources_per_layer"][1]

    full_shape = (num_sims, T, *grid_size)
    latent_shape = (num_sims, T_lat, *grid_size)
    full_chunk = (1, T, *grid_size)
    latent_chunk = (1, T_lat, *grid_size)

    for li in range(num_layers):
        lg = h5f.create_group(f"layer_{li}")

        for key in TRAIN_KEYS:
            lg.create_dataset(
                key,
                shape=full_shape,
                dtype=np.float16,
                chunks=full_chunk,
                compression="lzf",
            )

        for key in LATENT_KEYS:
            lg.create_dataset(
                key,
                shape=latent_shape,
                dtype=np.float16,
                chunks=latent_chunk,
                compression="lzf",
            )

    meta = h5f.create_group("meta")
    meta.attrs["config"] = json.dumps(cfg)
    meta.attrs["latent_downsample"] = latent_downsample
    meta.attrs["T_full"] = T
    meta.attrs["T_latent"] = T_lat

    pulse_type = sc.get("pulse_type", "rect")
    n_pulse_params = 2 + len(_PULSE_EXTRA_PARAMS.get(pulse_type, ["width"]))

    sources = meta.create_group("sources")

    sources.create_dataset(
        "layer",
        shape=(num_sims, max_sources),
        dtype=np.int32,
        fillvalue=-1,
    )

    sources.create_dataset(
        "position",
        shape=(num_sims, max_sources, 2),
        dtype=np.int32,
        fillvalue=-1,
    )

    sources.create_dataset(
        "num_pulses",
        shape=(num_sims, max_sources),
        dtype=np.int32,
        fillvalue=0,
    )

    sources.create_dataset(
        "pulses",
        shape=(
            num_sims,
            max_sources,
            max_pulses,
            n_pulse_params,
        ),
        dtype=np.float64,
        fillvalue=np.nan,
    )

    meta.create_dataset(
        "num_sources",
        shape=(num_sims,),
        dtype=np.int32,
    )

    meta.create_dataset(
        "seed",
        shape=(num_sims,),
        dtype=np.int64,
    )


def write_sim(h5f: h5py.File, idx: int, results: dict) -> None:
    """
    Write simulation idx into the preallocated HDF5 file.
    """

    k = int(h5f["meta"].attrs["latent_downsample"])

    for layer_idx, layer_data in results["layers"].items():
        lg = h5f[f"layer_{layer_idx}"]

        for key in TRAIN_KEYS:
            arr = layer_data.get(key)
            if arr is not None:
                lg[key][idx] = np.asarray(arr, dtype=np.float16)

        for key in LATENT_KEYS:
            arr = layer_data.get(key)
            if arr is not None:
                lg[key][idx] = np.asarray(arr[::k], dtype=np.float16)

    meta = h5f["meta"]
    sources = meta["sources"]
    m = results["meta"]

    num_sources = len(m["source_layer"])

    meta["num_sources"][idx] = num_sources
    meta["seed"][idx] = m["seed"]

    sources["layer"][idx, :num_sources] = m["source_layer"]
    sources["position"][idx, :num_sources] = m["source_position"]

    for s, pulse_list in enumerate(m["pulses"]):
        pulse_array = np.asarray(pulse_list, dtype=np.float64)
        n = len(pulse_array)

        sources["num_pulses"][idx, s] = n

        if n > 0:
            sources["pulses"][idx, s, :n] = pulse_array


def _run_one(args: tuple[int, dict, int | None]) -> tuple[int, dict]:
    """Worker target: run a single simulation and return (index, results)."""
    idx, cfg, seed = args
    return idx, run_simulation(cfg, seed=seed)


@hydra.main(
    version_base=None,
    config_path=str(CONFIG_DIR / "simulation"),
    config_name="linear",
)
def main(cfg: DictConfig) -> None:
    # Convert to a plain dict once so all downstream functions receive native Python types.
    cfg_dict: dict = OmegaConf.to_container(cfg, resolve=True)  # type: ignore

    # Resolve physical PSF settings (bold.voxel_mm, bold.psf_fwhm_mm) into the
    # per-layer grid-unit list consumed by simulate/train code and stored in the
    # HDF5 meta config (see _derive_psf_fwhm).
    cfg_dict["bold"]["psf_fwhm"] = _derive_psf_fwhm(
        cfg_dict["bold"], cfg_dict["simulation"]["num_layers"]
    )

    num_sims: int = cfg_dict.get("num_simulations", 1)
    output_path: str = cfg_dict.get("output_path", "simulations.h5")
    base_seed: int | None = cfg_dict.get("seed", 42)
    num_workers: int = cfg_dict.get("workers") or os.cpu_count() or 4
    latent_downsample: int = cfg_dict.get("latent_downsample", 1)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    seeds = [base_seed + i if base_seed is not None else None for i in range(num_sims)]
    tasks = [(i, cfg_dict, seeds[i]) for i in range(num_sims)]

    with h5py.File(output_path, "w") as h5f:
        init_h5(h5f, cfg_dict, num_sims, latent_downsample=latent_downsample)

        if num_workers == 1:
            for i, cfg_, seed in tasks:
                print(f"  [{i + 1}/{num_sims}] seed={seed}", flush=True)
                results = run_simulation(cfg_, seed=seed)
                write_sim(h5f, i, results)
        else:
            with ProcessPoolExecutor(max_workers=num_workers) as pool:
                futures = {pool.submit(_run_one, t): t[0] for t in tasks}
                for done, f in enumerate(as_completed(futures), 1):
                    idx, results = f.result()
                    write_sim(h5f, idx, results)
                    print(f"  [{done}/{num_sims}] seed={seeds[idx]}", flush=True)

    print(f"Saved {num_sims} simulation(s) -> {output_path}")


if __name__ == "__main__":
    main()
