from dataclasses import dataclass, field
from typing import Literal

import pandas as pd
from lightning.pytorch import LightningDataModule
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from walkjump.constants import ALPHABET_AHO
import numpy as np
from ._batch import HERWithLabelsBatch
from ._dataset import HERWithLabelDataset
import wandb


@dataclass
class HERWithLabelDataModule(LightningDataModule):
    def __init__(self, csv_data_path: str, batch_size: int = 64, num_workers: int = 1):
        self.csv_data_path = csv_data_path
        self.batch_size = batch_size    
        self.num_workers = num_workers
        self.dataset: pd.DataFrame = field(init=False)
        # self.alphabet: LabelEncoder = field(init=False, default=ALPHABET_AHO)
        self.alphabet: LabelEncoder = ALPHABET_AHO
        super().__init__()

    def prepare_data(self):
        pass
    
    def setup(self, stage: str):
        # print the current directory for debugging
        import os
        wandb.log({"cwd" : os.getcwd()})
        match stage:
            case "fit" | "validate" | "test":
                self.dataset = pd.read_csv(self.csv_data_path)

            case _:
                raise ValueError(f"Unreognized 'stage': {stage}")
        wandb.log({"dataset LENGTH": len(self.dataset)})

    def _make_dataloader(self, partition: Literal["train", "val", "test"]) -> DataLoader:
        df = self.dataset[self.dataset.split == partition]
        dataset = HERWithLabelDataset(df, self.alphabet)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=partition == "train",
            num_workers=self.num_workers,
            collate_fn=HERWithLabelsBatch.from_tensor_pylist,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_dataloader("train")

    def val_dataloader(self) -> DataLoader:
        return self._make_dataloader("val")

    def test_dataloader(self) -> DataLoader:
        return self._make_dataloader("test")