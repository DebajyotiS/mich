from torch.utils.data import Dataset
from dataclasses import dataclass
from pathlib import Path
from pytorch_lightning.core.datamodule import LightningDataModule


@dataclass
class SynethicDataLoadConfig:
    path: str | Path


def synthetic_data_loader(load_config: SynethicDataLoadConfig) -> np.ndarray:
    pass  # Placeholder for future implementation


class SyntheticDataset(Dataset):
    def __init__(self, dataset_config: datasetConfig) -> None:
        pass


class SyntheticDataModule(LightningDataModule):
    def __init__(self, data_module_config: dataModuleConfig) -> None:
        super().__init__()
        pass

    def setup(self, stage: str | None = None) -> None:
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass
