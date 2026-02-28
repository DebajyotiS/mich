import logging

import hydra
import pytorch_lightning as pl

import torch
from omegaconf import DictConfig, ListConfig


from src.utils.hydra_utils import (
    instantiate_collection,
    log_hyperparameters,
    print_config,
    reload_original_config,
    save_config,
)
import rootutils
root = rootutils.setup_root(__file__, pythonpath=True, cwd=False)
log = logging.getLogger(__name__)


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
    else:
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
    # Only initiate on main rank

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
    import os

    # Set environment variable to show full error tracebacks in Hydra
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    main()
    log.info("All done. Exiting gracefully.")
