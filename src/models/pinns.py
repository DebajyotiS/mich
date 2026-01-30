from torch import nn
from typing import Mapping
from pytorch_lightning import LightningModule


class PINNModel(nn.Module):
    def __init__(self, config: Mapping) -> None:
        self.config = config
        super().__init__()


class MICH(LightningModule):
    def __init__(self, config: Mapping) -> None:
        self.config = config
        super().__init__()
        pass

    def _shared_step(self):
        pass

    def training_step(self):
        pass

    def validation_step(self):
        pass

    def on_validation_epoch_end(self) -> None:
        pass

    def configure_optimizers(self):
        pass
