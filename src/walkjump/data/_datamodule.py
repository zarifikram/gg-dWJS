from dataclasses import dataclass, field
from typing import Literal

import pandas as pd
from lightning.pytorch import LightningDataModule
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from walkjump.constants import ALPHABET_AHO, ALPHABET_AMP
import numpy as np
from ._batch import AMPWithLabelsBatch, AbBatch, AbWithLabelBatch, PCAbWithLabelBatch, AMPBatch
from ._dataset import AMPWithLabelDataset, AbDataset, AbWithLabelDataset, PCAbWithLabelDataset, AMPDataset
import wandb

@dataclass
class AbDataModule(LightningDataModule):
    def __init__(self, csv_data_path: str, batch_size: int = 64, num_workers: int = 1):
        self.csv_data_path = csv_data_path
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
        import os
        wandb.log({"cwd" : os.getcwd()})
        match stage:
            case "fit" | "validate" | "test":
                self.dataset = pd.read_csv(self.csv_data_path, compression="gzip")
            case _:
                raise ValueError(f"Unreognized 'stage': {stage}")
        wandb.log({"dataset LENGTH": len(self.dataset)})

    def _make_dataloader(self, partition: Literal["train", "val", "test"]) -> DataLoader:
        df = self.dataset[self.dataset.partition == partition]
        dataset = AbDataset(df, self.alphabet)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=partition == "train",
            num_workers=self.num_workers,
            collate_fn=AbBatch.from_tensor_pylist,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_dataloader("train")

    def val_dataloader(self) -> DataLoader:
        return self._make_dataloader("val")

    def test_dataloader(self) -> DataLoader:
        return self._make_dataloader("test")

@dataclass
class AbWithLabelDataModule(LightningDataModule):
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
                self.dataset = pd.read_csv(self.csv_data_path, compression="gzip")

            case _:
                raise ValueError(f"Unreognized 'stage': {stage}")
        wandb.log({"dataset LENGTH": len(self.dataset)})

    def _make_dataloader(self, partition: Literal["train", "val", "test"]) -> DataLoader:
        df = self.dataset[self.dataset.type == partition]
        dataset = AbWithLabelDataset(df, self.alphabet)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=partition == "train",
            num_workers=self.num_workers,
            collate_fn=AbWithLabelBatch.from_tensor_pylist,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_dataloader("train")

    def val_dataloader(self) -> DataLoader:
        return self._make_dataloader("val")

    def test_dataloader(self) -> DataLoader:
        return self._make_dataloader("test")

@dataclass
class PCAbWithLabelDataModule(LightningDataModule):
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
                # self.dataset.instability_index = (self.dataset.instability_index - self.dataset.instability_index.mean()) / self.dataset.instability_index.std()
                # self.dataset.aromaticity = (self.dataset.aromaticity - self.dataset.aromaticity.mean()) / self.dataset.aromaticity.std()
                # self.dataset.ss_perc_sheet = (self.dataset.ss_perc_sheet - self.dataset.ss_perc_sheet.mean()) / self.dataset.ss_perc_sheet.std()
                # give random weights for these two columns w1 for instability_index and w2 for ss_perc_sheet
                # self.dataset['w1'] = np.random.random(len(self.dataset))
                # self.dataset['w2'] = 1 - self.dataset['w1']
                # self.dataset['single_value'] = self.dataset.instability_index * self.dataset.w1 + self.dataset.ss_perc_sheet * self.dataset.w2
                # self.dataset['single_value'] = (self.dataset['single_value'] - self.dataset['single_value'].mean()) / self.dataset['single_value'].std()


            case _:
                raise ValueError(f"Unreognized 'stage': {stage}")
        wandb.log({"dataset LENGTH": len(self.dataset)})

    def _make_dataloader(self, partition: Literal["train", "val", "test"]) -> DataLoader:
        df = self.dataset[self.dataset.type == partition]
        dataset = PCAbWithLabelDataset(df, self.alphabet)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=partition == "train",
            num_workers=self.num_workers,
            collate_fn=PCAbWithLabelBatch.from_tensor_pylist,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_dataloader("train")

    def val_dataloader(self) -> DataLoader:
        return self._make_dataloader("val")

    def test_dataloader(self) -> DataLoader:
        return self._make_dataloader("test")


@dataclass
class AMPDataModule(LightningDataModule):
    def __init__(self, csv_data_path: str, batch_size: int = 64, num_workers: int = 1):
        self.csv_data_path = csv_data_path
        self.batch_size = batch_size    
        self.num_workers = num_workers
        self.dataset: pd.DataFrame = field(init=False)
        # self.alphabet: LabelEncoder = field(init=False, default=ALPHABET_AHO)
        self.alphabet: LabelEncoder = ALPHABET_AMP
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
        dataset = AMPDataset(df, self.alphabet)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=partition == "train",
            num_workers=self.num_workers,
            collate_fn=AMPBatch.from_tensor_pylist,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_dataloader("train")

    def val_dataloader(self) -> DataLoader:
        return self._make_dataloader("valid")

    def test_dataloader(self) -> DataLoader:
        return self._make_dataloader("valid")
    
    
# train_data_path: "data/amp/train.pkl"
# validation_data_path: "data/amp/valid.pkl"



@dataclass
class AMPWithLabelsDataModule(LightningDataModule):
    def __init__(self, train_data_path: str, validation_data_path: str, batch_size: int = 64, num_workers: int = 1):
        self.train_data_path = train_data_path
        self.validation_data_path = validation_data_path
        self.batch_size = batch_size    
        self.num_workers = num_workers
        self.dataset: pd.DataFrame = field(init=False)
        # self.alphabet: LabelEncoder = field(init=False, default=ALPHABET_AHO)
        self.alphabet: LabelEncoder = ALPHABET_AMP
        super().__init__()

    def prepare_data(self):
        pass
    
    def setup(self, stage: str):
        # print the current directory for debugging
        import os
        wandb.log({"cwd" : os.getcwd()})
        match stage:
            case "fit" | "validate" | "test":
                with open(self.train_data_path, "rb") as f:
                    self.train_dict = pd.read_pickle(f)
                
                with open(self.validation_data_path, "rb") as f:
                    self.validation_dict = pd.read_pickle(f)
            case _:
                raise ValueError(f"Unreognized 'stage': {stage}")
        wandb.log({"dataset LENGTH": len(self.train_dict["prediction"]) + len(self.validation_dict["prediction"])})

    def _make_dataloader(self, partition: Literal["train", "val", "test"]) -> DataLoader:
        if partition == "train":
            df = self.train_dict
        elif partition == "val":
            df = self.validation_dict
        dataset = AMPWithLabelDataset(df, self.alphabet)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=partition == "train",
            num_workers=self.num_workers,
            collate_fn=AMPWithLabelsBatch.from_tensor_pylist,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_dataloader("train")

    def val_dataloader(self) -> DataLoader:
        return self._make_dataloader("val")

    def test_dataloader(self) -> DataLoader:
        return self._make_dataloader("val")