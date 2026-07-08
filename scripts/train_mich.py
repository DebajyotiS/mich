import logging
import os
import subprocess

import h5py
import hydra
import pytorch_lightning as pl
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

from mich import CONFIG_DIR
from mich.data.synthetic import compute_split_counts, discover_layers
from mich.models.blocks import HEINZLE_SIGNALS, HEINZLE_SIGNALS_SINGLE
from mich.utils.hydra_utils import (
    instantiate_collection,
    log_hyperparameters,
    print_config,
    reload_original_config,
    save_config,
)

log = logging.getLogger(__name__)


def _resolve_data_layers(cfg: DictConfig) -> tuple[str, ...]:
    """Mirror SyntheticDataModule._make_dataset's layer resolution without instantiating it."""
    layers_val = cfg.datamodule.data.get("layers", "auto")
    if layers_val is None or layers_val == "auto":
        return discover_layers(cfg.datamodule.data.path)
    return tuple(layers_val)


def _resolve_total_steps(cfg: DictConfig) -> int:
    """Mirror SyntheticDataModule's train-split + drop_last dataloader math, without
    instantiating the dataset, so scheduler.T_max can auto-track dataset size, split
    fraction, batch size, and max_epochs instead of needing to be hand-recomputed."""
    layers = _resolve_data_layers(cfg)
    with h5py.File(cfg.datamodule.data.path, "r") as f:
        n = int(f[layers[0]]["bold"].shape[0])
    n_train, _, _ = compute_split_counts(n, dict(cfg.datamodule.split))
    batch_size = int(cfg.datamodule.loader.batch_size)
    drop_last = bool(cfg.datamodule.loader.get("drop_last", True))
    steps_per_epoch = n_train // batch_size if drop_last else -(-n_train // batch_size)
    return steps_per_epoch * int(cfg.trainer.max_epochs)


def _inject_scheduler_steps(cfg: DictConfig) -> None:
    """Auto-set scheduler.T_max from the actual dataset/split/batch/epoch config, unless
    the user has pinned an explicit value (e.g. for a deliberately shorter warm-restart)."""
    if cfg.model.scheduler.get("T_max") is not None:
        return
    total_steps = _resolve_total_steps(cfg)
    with open_dict(cfg):
        cfg.model.scheduler.T_max = total_steps
    log.info(f"Auto-set scheduler.T_max = {total_steps} (from dataset size/split/batch/epochs)")


def _inject_sim_physics(cfg: DictConfig, sim_cfg: dict) -> None:
    """Overwrite model physics constants AND structural shape with values derived from
    the simulation HDF5, so the model architecture always matches the data it's pointed at.

    Derives k1/k2/k3 from the raw acquisition + haemodynamic parameters so the
    model always trains with constants that match the data it was generated from.
    Also derives L (number of cortical layers actually loaded by the datamodule),
    out_channels, and the Heinzle signal set (5 signals for single-layer/no-drainage
    data, 7 for multi-layer/drainage data) -- this is what lets a single model config
    work against any scenario file without hand-editing L/out_channels per run.
    """
    ac = sim_cfg["acquisition"]
    hc = sim_cfg["haemodynamic"]
    layers = sim_cfg["simulation"]["layers"]

    k1 = 4.3 * ac["f0"] * hc["E0"] * ac["TE"]
    k2 = ac["eps"] * ac["r0"] * hc["E0"] * ac["TE"]
    k3 = 1.0 - ac["eps"]

    # lambda_d for layer > 0 (deep layer drain is always 0)
    lambda_d = layers[1]["lambda_d"] if len(layers) > 1 else layers[0]["lambda_d"]

    num_layers = len(_resolve_data_layers(cfg))
    has_drain = num_layers > 1
    signals = HEINZLE_SIGNALS if has_drain else HEINZLE_SIGNALS_SINGLE

    with open_dict(cfg):
        cfg.model.haemo = {
            "kappa": hc["kappa"],
            "gamma": hc["gamma"],
            "alpha": hc["alpha"],
            "tau": layers[0]["tau"],
            "lambda_d": lambda_d,
            "tau_d": sim_cfg["simulation"]["tau_d"],
        }
        cfg.model.acquisition = {
            "f0": ac["f0"],
            "E0": hc["E0"],
            "TE": ac["TE"],
            "r0": ac["r0"],
            "eps": ac["eps"],
            "k1": k1,
            "k2": k2,
            "k3": k3,
        }
        cfg.model.V0 = hc["V0"]
        cfg.model.psf_fwhm = sim_cfg["bold"]["psf_fwhm"]
        # Physics-loss residual must use the same Balloon-Windkessel approximation the
        # data was actually generated with, or the physics loss evaluates the wrong ODE.
        cfg.model.loss_config.order = sim_cfg["simulation"]["order"]
        cfg.model.L = num_layers
        # out_channels == len(signals), NOT multiplied by L -- SpatioTemporalDecoder
        # builds len(signals)*L output heads internally, using L separately (blocks.py).
        cfg.model.out_channels = len(signals)
        cfg.model.heinzle_net.spatial_decoder_config.signals = list(signals)


@hydra.main(
    version_base=None,
    config_path=str(CONFIG_DIR),
    config_name="mainconfig.yaml",
)
def main(cfg: DictConfig) -> None:
    log.info("Setting up job configuration")
    if cfg.resume.state:
        log.info("Reloading original config and resuming training")
        # Re-apply this invocation's own CLI overrides (e.g. trainer.max_epochs=800) on
        # top of the reloaded original config -- otherwise reload_original_config's
        # wholesale replacement of cfg would silently discard anything passed on the
        # command line for this resume run beyond network_name/project_name/resume.state.
        overrides_task = HydraConfig.get().overrides.task
        cli_overrides = OmegaConf.from_dotlist(overrides_task)
        orig_cfg = reload_original_config(
            path=cfg.paths.full_path,
            ckpt_flag=cfg.resume.ckpt_flag,
            reload_states=False,
        )  # type: ignore
        cfg = OmegaConf.merge(orig_cfg, cli_overrides)

        # T_max was resolved against the ORIGINAL run's max_epochs and is now a concrete
        # (non-None) value inherited via the merge above -- _inject_scheduler_steps's own
        # "leave alone if already set" guard means it would never recompute this, so
        # extending trainer.max_epochs on resume would otherwise leave the cosine
        # schedule exhausted (sitting at eta_min) for the entire extension. Recompute it
        # against whatever max_epochs applies to this invocation, unless the user
        # explicitly pinned scheduler.T_max themselves in this invocation's own overrides
        # (matching _inject_scheduler_steps's documented pin-override intent).
        user_pinned_t_max = any(
            o.split("=")[0] == "model.scheduler.T_max" for o in overrides_task
        )
        if not user_pinned_t_max:
            with open_dict(cfg):
                cfg.model.scheduler.T_max = None
            _inject_scheduler_steps(cfg)

    if cfg.model.loss_config.lambda_supervision == 0.0:
        log.info("lambda_supervision=0, disabling return_latents in datamodule")
        cfg.datamodule.data.return_latents = False

    log.info("Instantiating datamodule")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    if not cfg.resume.state:
        log.info("Injecting physics constants from simulation HDF5")
        _inject_sim_physics(cfg, datamodule.sim_config)
        _inject_scheduler_steps(cfg)

    if cfg.print_config:
        log.info("Printing configuration")
        print_config(cfg)

    if cfg.seed:
        log.info(f"Setting seed to {cfg.seed}")
        pl.seed_everything(cfg.seed)

    if cfg.precision:
        log.info(f"Setting matrix precision to {cfg.precision}")
        torch.set_float32_matmul_precision(cfg.precision)

    if getattr(cfg.model.loss_config, "lambda_supervision", 0.0) == 0.0:
        log.info("lambda_supervision=0, disabling return_latents in datamodule")
        cfg.datamodule.data.return_latents = False

    log.info("Instantiating datamodule")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    log.info("Instantiating model")
    model = hydra.utils.instantiate(
        cfg.model,
    )
    if cfg.compile:
        if torch.backends.mps.is_available():
            log.warning(
                "Currently using MPS. Compiling a torch model would not work, as Triton has no support for it."
            )
        else:
            log.info("Compiling model")
            model = torch.compile(model, mode=cfg.compile)

    log.info("Instantiating callbacks")
    callbacks = instantiate_collection(cfg.callbacks)

    log.info("Instantiating loggers")
    loggers = instantiate_collection(cfg.loggers)

    log.info("Instantiating trainer")
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    if loggers:
        log.info("Logging hyperparameters")
        log_hyperparameters(cfg, model, trainer)  # type: ignore

    # Saved after loggers are instantiated (not right after config setup) so that, when
    # wandb is active, save_config's `cfg.loggers.wandb.id = wandb.run.id` capture has a
    # real run to record -- doing this earlier always saw wandb.run as None, since the
    # logger didn't exist yet, silently breaking wandb continuity for any later resume.
    # Runs unconditionally (not just for fresh runs) so a resumed run's saved config
    # reflects whatever values were actually used this time (e.g. an extended
    # trainer.max_epochs), not the original run's values.
    log.info("Saving configuration")
    save_config(cfg)

    if cfg.train:
        log.info("MICH training started")
        trainer.fit(model, datamodule, ckpt_path=cfg.ckpt_path)
    log.info("MICH training complete")


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"

    if torch.backends.mps.is_available():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        print("Using Apple Metal GPU (mps)")

    elif torch.cuda.is_available():
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        if not os.environ.get("CUDA_VISIBLE_DEVICES"):
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                rows = [
                    line.split(",") for line in result.stdout.strip().split("\n") if line.strip()
                ]
                best = min(rows, key=lambda r: int(r[1]))
                selected = best[0].strip()
                os.environ["CUDA_VISIBLE_DEVICES"] = selected
                print(f"Auto-selected GPU {selected}")
            else:
                log.warning("nvidia-smi failed, not setting CUDA_VISIBLE_DEVICES")

        print("Using CUDA GPU")

    else:
        print("Using CPU")

    main()
    log.info("All done. Exiting gracefully.")
