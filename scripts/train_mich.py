import logging
import os
import subprocess

import hydra
import pytorch_lightning as pl
import rootutils
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from src.utils.hydra_utils import (
    instantiate_collection,
    log_hyperparameters,
    print_config,
    reload_original_config,
    save_config,
)

root = rootutils.setup_root(__file__, pythonpath=True, cwd=False)
log = logging.getLogger(__name__)


def _inject_sim_physics(cfg: DictConfig, sim_cfg: dict) -> None:
    """Overwrite model physics constants with values from the simulation HDF5.

    Derives k1/k2/k3 from the raw acquisition + haemodynamic parameters so the
    model always trains with constants that match the data it was generated from.
    """
    ac = sim_cfg["acquisition"]
    hc = sim_cfg["haemodynamic"]
    layers = sim_cfg["simulation"]["layers"]

    k1 = 4.3 * ac["f0"] * hc["E0"] * ac["TE"]
    k2 = ac["eps"] * ac["r0"] * hc["E0"] * ac["TE"]
    k3 = 1.0 - ac["eps"]

    # lambda_d for layer > 0 (deep layer drain is always 0)
    lambda_d = layers[1]["lambda_d"] if len(layers) > 1 else layers[0]["lambda_d"]

    with open_dict(cfg):
        cfg.model.haemo = {
            "kappa": hc["kappa"],
            "gamma": hc["gamma"],
            "alpha": hc["alpha"],
            "tau": layers[0]["tau"],
            "lambda_d": lambda_d,
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


@hydra.main(
    version_base=None,
    config_path=str(root / "config"),
    config_name="mainconfig.yaml",
)
def main(cfg: DictConfig) -> None:
    log.info("Setting up job configuration")
    if cfg.resume.state:
        log.info("Reloading original config and resuming training")
        cfg = reload_original_config(cfg, reload_states=False)  # type: ignore

    if cfg.model.loss_config.lambda_supervision == 0.0:
        log.info("lambda_supervision=0, disabling return_latents in datamodule")
        cfg.datamodule.data.return_latents = False

    log.info("Instantiating datamodule")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    if not cfg.resume.state:
        log.info("Injecting physics constants from simulation HDF5")
        _inject_sim_physics(cfg, datamodule.sim_config)
        log.info("Saving configuration")
        save_config(cfg)

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
