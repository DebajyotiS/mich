from __future__ import annotations

import argparse
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import yaml

from src.data.balloon import (
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
from src.data.neuronal import LayeredDiffusionSimulator, NeuralSimulatorParams
from src.data.signals import Noise, Pulse, Sources


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


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


def run_simulation(cfg: dict, seed: int | None = None) -> dict:
    """Run one complete simulation and return a results dict."""
    rng = np.random.default_rng(seed)
    sc = cfg["simulation"]

    num_layers: int = sc["num_layers"]
    grid_size = tuple(sc["grid_size"])
    dt: float = sc["dt"]
    time_duration: int = sc["time_duration"]
    max_pulses: int = sc["max_pulses"]
    steps: int = int(time_duration / dt)
    order: str = sc["order"]

    layers_cfg: list[dict] = sc["layers"]
    if len(layers_cfg) != num_layers:
        raise ValueError(f"Config has {len(layers_cfg)} layer entries but num_layers={num_layers}")

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

    num_pulses = int(rng.integers(1, max_pulses + 1))
    durations = rng.uniform(2.0, 10.0, size=num_pulses)
    amplitudes = rng.uniform(0.3, 1.0, size=num_pulses)
    isi_min: int = sc.get("isi_min", 20)  # jitter window (s) for random onset placement

    # Build non-overlapping onsets sequentially: neural pulses never overlap,
    # but BOLD responses may (haemodynamic overlap is fine).
    onsets = []
    t_min = 10
    for k in range(num_pulses):
        onset = t_min + int(rng.integers(0, isi_min))
        onsets.append(onset)
        t_min = onset + int(np.ceil(durations[k])) + 1  # +1 s gap: no neural overlap
    onsets = np.array(onsets)

    pulse_list = [
        [float(a), float(o), float(d)]
        for a, o, d in zip(amplitudes, onsets, durations, strict=True)
    ]
    pulse = Pulse(
        pulse_type=sc.get("pulse_type", "rect"),
        peaks=pulse_list,
        duration=time_duration,
        dt=dt,
    )

    # Source placement
    sources = Sources()
    source_layer = int(rng.integers(0, num_layers))
    source_pos = tuple(rng.integers(0, grid_size[0], size=2).tolist())
    sources.add_source(layer=source_layer, position=source_pos, signal=pulse.generate()[1])

    # Neural diffusion
    neural_noise = Noise(type="pink", seed=seed, domain="both")
    simulator = LayeredDiffusionSimulator(sim_params)
    history = simulator.simulate(
        sources=sources.get_sources(),
        noise=neural_noise,
        steps=steps,
        snr_db=float(sc["neural_SNR"]),
    )

    x_inputs = [history[:, i, ...] for i in range(num_layers)]

    # Upsample neural activity to haemodynamic integration resolution (zero-order hold)
    haemo_dt: float = sc.get("haemo_dt", dt)
    haemo_ratio = max(1, round(dt / haemo_dt))
    if haemo_ratio > 1:
        x_inputs_haemo = [np.repeat(xi, haemo_ratio, axis=0) for xi in x_inputs]
    else:
        x_inputs_haemo = x_inputs

    # Cortex layers (bottom-up)
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

    # Haemodynamic simulation
    out = simulate_cortex(
        cortex_layers,
        haemo,
        x_inputs=x_inputs_haemo,  # type: ignore
        dt=haemo_dt,
        tau_d=sc["tau_d"],
        order=order
    )

    # Downsample haemodynamic outputs back to neural dt resolution for storage
    if haemo_ratio > 1:
        out = {
            li: {k: (v[::haemo_ratio] if v is not None else None) for k, v in ld.items()}
            for li, ld in out.items()
        }

    # BOLD readout
    bold_cfg = cfg.get("bold", {})
    psf_fwhm = bold_cfg.get("psf_fwhm", [18.0, 27.675, 40.05])
    if len(psf_fwhm) != num_layers:
        raise ValueError(f"psf_fwhm has {len(psf_fwhm)} entries but num_layers={num_layers}")

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
            out[i],
            acq,
            haemo,
            layer_depth=i,
            params=bold_params,
        )
        for i in range(num_layers)
    }

    # Pack results
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


def init_h5(h5f: h5py.File, cfg: dict, num_sims: int, latent_downsample: int = 10) -> None:
    """
    Pre-allocate all datasets for *num_sims* simulations.

    Training keys (x, bold):
        Shape  : (N, T, H, W)  at full resolution
        Chunks : (1, T, H, W)  -- one sim per chunk, contiguous in time

    Latent keys (s, f, v, q, v_star, q_star):
        Shape  : (N, T//latent_downsample, H, W)  at reduced resolution
        Chunks : (1, T//latent_downsample, H, W)

    Args:
        latent_downsample : int, store one latent frame every k timesteps.
                            At dt=0.5s and k=10, that is one frame every 5s --
                            sufficient to inspect haemodynamic dynamics whose
                            timescale is 5-30s.
    """
    if not isinstance(latent_downsample, int) or latent_downsample < 1:
        raise ValueError(f"latent_downsample must be a positive int, got {latent_downsample!r}")

    sc = cfg["simulation"]
    num_layers = sc["num_layers"]
    grid_size = tuple(sc["grid_size"])
    T = int(sc["time_duration"] / sc["dt"])
    T_lat = T // latent_downsample
    max_pulses = sc["max_pulses"]

    full_shape = (num_sims, T, *grid_size)
    latent_shape = (num_sims, T_lat, *grid_size)
    full_chunk = (1, T, *grid_size)
    latent_chunk = (1, T_lat, *grid_size)

    for li in range(num_layers):
        lg = h5f.create_group(f"layer_{li}")

        for key in TRAIN_KEYS:
            lg.create_dataset(
                key, shape=full_shape, dtype=np.float16, chunks=full_chunk, compression="lzf"
            )

        for key in LATENT_KEYS:
            lg.create_dataset(
                key, shape=latent_shape, dtype=np.float16, chunks=latent_chunk, compression="lzf"
            )

    meta = h5f.create_group("meta")
    meta.attrs["config"] = json.dumps(cfg)
    meta.attrs["latent_downsample"] = latent_downsample  # int, readable by write_sim
    meta.attrs["T_full"] = T
    meta.attrs["T_latent"] = T_lat

    # pulses: keep float64 -- onset times need sub-TR precision
    meta.create_dataset(
        "pulses", shape=(num_sims, max_pulses, 3), dtype=np.float64, fillvalue=np.nan
    )
    meta.create_dataset("num_pulses", shape=(num_sims,), dtype=np.int32)
    meta.create_dataset("source_layer", shape=(num_sims,), dtype=np.int32)
    meta.create_dataset("source_position", shape=(num_sims, 2), dtype=np.int32)
    meta.create_dataset("seed", shape=(num_sims,), dtype=np.int64)


def write_sim(h5f: h5py.File, idx: int, results: dict) -> None:
    """
    Write simulation *idx* into the pre-allocated datasets.

    Latent keys are downsampled along the time axis on write so no extra
    memory is needed relative to what run_simulation already allocated.
    """
    k = int(h5f["meta"].attrs["latent_downsample"])  # type: ignore

    for layer_idx, layer_data in results["layers"].items():
        lg = h5f[f"layer_{layer_idx}"]

        for key in TRAIN_KEYS:
            arr = layer_data.get(key)
            if arr is not None:
                lg[key][idx] = np.asarray(arr, dtype=np.float16)  # type: ignore

        for key in LATENT_KEYS:
            arr = layer_data.get(key)
            if arr is not None:
                lg[key][idx] = np.asarray(arr[::k], dtype=np.float16)  # type: ignore

    meta = h5f["meta"]
    m = results["meta"]
    pulses = np.asarray(m["pulses"], dtype=np.float64)
    meta["pulses"][idx, : len(pulses)] = pulses  # type: ignore
    meta["num_pulses"][idx] = m["num_pulses"]  # type: ignore
    meta["source_layer"][idx] = m["source_layer"]  # type: ignore
    meta["source_position"][idx] = m["source_position"]  # type: ignore
    meta["seed"][idx] = m["seed"]  # type: ignore


def _run_one(args: tuple[int, dict, int | None]) -> tuple[int, dict]:
    """Worker target: run a single simulation and return (index, results)."""
    idx, cfg, seed = args
    return idx, run_simulation(cfg, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BOLD simulations from a YAML config")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output HDF5 path")
    parser.add_argument("-n", "--num-sims", type=int, default=None, help="Override num_simulations")
    parser.add_argument(
        "-w", "--workers", type=int, default=None, help="Parallel workers (default: CPU count)"
    )
    parser.add_argument(
        "--latent-downsample",
        type=int,
        default=1,
        help="Downsample factor for latent states (default: 1)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    num_sims = args.num_sims or cfg.get("num_simulations", 1)
    output_path = args.output or cfg.get("output_path", "simulations.h5")
    base_seed = cfg.get("seed", 42)
    num_workers = args.workers or os.cpu_count()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    seeds = [base_seed + i if base_seed is not None else None for i in range(num_sims)]
    tasks = [(i, cfg, seeds[i]) for i in range(num_sims)]

    with h5py.File(output_path, "w") as h5f:
        init_h5(h5f, cfg, num_sims, latent_downsample=args.latent_downsample)

        if num_workers == 1:
            for i, cfg_, seed in tasks:
                print(f"  [{i + 1}/{num_sims}] seed={seed}", flush=True)
                results = run_simulation(cfg_, seed=seed)
                write_sim(h5f, i, results)
        else:
            done = 0
            task_iter = iter(tasks)
            with ProcessPoolExecutor(max_workers=num_workers) as pool:
                # Seed pool with min(num_workers, num_sims) tasks to avoid
                # submitting more futures than there are simulations
                pending: dict = {}
                for t in itertools.islice(task_iter, num_workers):
                    f = pool.submit(_run_one, t)
                    pending[f] = t[0]

                while pending:
                    finished = next(iter(as_completed(pending)))
                    idx, results = finished.result()
                    write_sim(h5f, idx, results)
                    del pending[finished]
                    done += 1
                    print(f"  [{done}/{num_sims}] seed={seeds[idx]}", flush=True)

                    nxt = next(task_iter, None)
                    if nxt is not None:
                        f = pool.submit(_run_one, nxt)
                        pending[f] = nxt[0]

    print(f"Saved {num_sims} simulation(s) -> {output_path}")


if __name__ == "__main__":
    main()
