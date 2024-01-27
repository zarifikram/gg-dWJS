from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from lightning.pytorch import LightningDataModule
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from walkjump.constants import ALPHABET_AHO

from ._batch import MNISTBatch, MNISTWithLabelsBatch
from ._dataset import MNISTDataset, MNISTWithLabelsDataset
import wandb

@dataclass
class MNISTDataModule(LightningDataModule):
    def __init__(self, train_data_path: str, test_data_path: str, val_data_path: str, batch_size: int = 64, num_workers: int = 1):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.val_data_path = val_data_path
        self.batch_size = batch_size    
        self.num_workers = num_workers
        self.dataset: pd.DataFrame = field(init=False)
        # self.alphabet: LabelEncoder = field(init=False, default=ALPHABET_AHO)
        self.alphabet: LabelEncoder = ALPHABET_AHO
        super().__init__()
    # csv_data_path: str
    # batch_size: int = 64
    # num_workers: int = 1

    # dataset: pd.DataFrame = field(init=False)
    # alphabet: LabelEncoder = field(init=False, default=ALPHABET_AHO)

    def prepare_data(self):
        pass
    
    def setup(self, stage: str):
        # print the current directory for debugging
        pass
        # import os
        # wandb.log({"cwd" : os.getcwd()})
        # match stage:
        #     case "fit" | "validate" | "test":
        #         self.dataset = pd.read_csv(self.csv_data_path, compression="gzip")
        #     case _:
        #         raise ValueError(f"Unreognized 'stage': {stage}")
        # wandb.log({"dataset LENGTH": len(self.dataset)})

    def _make_dataloader(self, partition: Literal["train", "val", "test"]) -> DataLoader:
        data = self._get_data(partition)
        data = MNISTDataset(data)
        
        return DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=partition == "train",
            num_workers=self.num_workers,
            collate_fn=MNISTBatch.from_tensor_pylist,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_dataloader("train")

    def val_dataloader(self) -> DataLoader:
        return self._make_dataloader("val")

    def test_dataloader(self) -> DataLoader:
        return self._make_dataloader("test")

    def _get_data(self, partition: Literal["train", "val", "test"]) -> np.ndarray:
        partition_to_dir = {"train" : self.train_data_path, "val" : self.val_data_path, "test" : self.test_data_path}
        with open(partition_to_dir[partition]) as f:
            lines = f.readlines()
        arr = np.array([[int(i) for i in line.split()] for line in lines])
        # np.random.shuffle(arr)
        return arr


@dataclass
class MNISTWithLabelsDataModule(LightningDataModule):
    def __init__(self, data_path: str, batch_size: int = 64, num_workers: int = 1):
        self.data_path = data_path
        self.batch_size = batch_size    
        self.num_workers = num_workers
        self.dataset: pd.DataFrame = field(init=False)
        # self.alphabet: LabelEncoder = field(init=False, default=ALPHABET_AHO)
        self.alphabet: LabelEncoder = ALPHABET_AHO
        super().__init__()
    # csv_data_path: str
    # batch_size: int = 64
    # num_workers: int = 1

    # dataset: pd.DataFrame = field(init=False)
    # alphabet: LabelEncoder = field(init=False, default=ALPHABET_AHO)

    def prepare_data(self):
        pass
    
    def setup(self, stage: str):
        # print the current directory for debugging
        pass

    def _make_dataloader(self, partition: Literal["train", "val", "test"]) -> DataLoader:
        data, labels = self._get_data(partition)
        data = MNISTWithLabelsDataset(data, labels)
        
        return DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=partition == "train",
            num_workers=self.num_workers,
            collate_fn=MNISTWithLabelsBatch.from_tensor_pylist,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_dataloader("train")

    def val_dataloader(self) -> DataLoader:
        return self._make_dataloader("val")

    def test_dataloader(self) -> DataLoader:
        return self._make_dataloader("test")

    def _get_data(self, partition: Literal["train", "val", "test"]) -> tuple[np.ndarray, np.ndarray]:
        partition_to_data = {"train" : "train.npy", "val" : "val.npy", "test" : "test.npy"}
        partition_to_labels = {"train" : "binarized_train_labels.npy", "val" : "binarized_train_labels.npy", "test" : "binarized_test_labels.npy"}
        
        data = np.load(self.data_path + partition_to_data[partition])
        labels = np.load(self.data_path + partition_to_labels[partition])
        if partition in ['train']:
            labels = labels[:50000]
        elif partition in ['val']:
            labels = labels[50000:]
        assert len(data) == len(labels)
        return data, labels

