from dataclasses import InitVar, dataclass, field

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from walkjump.constants import ALPHABET_AHO
from walkjump.utils import token_string_to_tensor


@dataclass
class AbDataset(Dataset):
    df: pd.DataFrame
    alphabet_or_token_list: InitVar[LabelEncoder | list[str]] = ALPHABET_AHO
    alphabet: LabelEncoder = field(init=False)

    def __post_init__(self, alphabet_or_token_list: LabelEncoder | list[str]):
        self.alphabet = (
            alphabet_or_token_list
            if isinstance(alphabet_or_token_list, LabelEncoder)
            else LabelEncoder().fit(alphabet_or_token_list)
        )
        self.df.reset_index(drop=True, inplace=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> torch.Tensor:
        row = self.df.loc[index]
        tensor_h = token_string_to_tensor(row.fv_heavy_aho, self.alphabet)
        tensor_l = token_string_to_tensor(row.fv_light_aho, self.alphabet)
        return torch.cat([tensor_h, tensor_l])

@dataclass
class AbWithLabelDataset(Dataset):
    df: pd.DataFrame
    alphabet_or_token_list: InitVar[LabelEncoder | list[str]] = ALPHABET_AHO
    alphabet: LabelEncoder = field(init=False)

    def __post_init__(self, alphabet_or_token_list: LabelEncoder | list[str]):
        self.alphabet = (
            alphabet_or_token_list
            if isinstance(alphabet_or_token_list, LabelEncoder)
            else LabelEncoder().fit(alphabet_or_token_list)
        )
        self.df.reset_index(drop=True, inplace=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> torch.Tensor:
        row = self.df.loc[index]
        tensor_h = token_string_to_tensor(row.HeavyAA_aligned, self.alphabet)
        tensor_l = token_string_to_tensor(row.LightAA_aligned, self.alphabet)
        label = torch.tensor(row.single_value).float()

        return torch.cat([tensor_h, tensor_l]), label

@dataclass
class PCAbWithLabelDataset(Dataset):
    df: pd.DataFrame
    alphabet_or_token_list: InitVar[LabelEncoder | list[str]] = ALPHABET_AHO
    alphabet: LabelEncoder = field(init=False)

    def __post_init__(self, alphabet_or_token_list: LabelEncoder | list[str]):
        self.alphabet = (
            alphabet_or_token_list
            if isinstance(alphabet_or_token_list, LabelEncoder)
            else LabelEncoder().fit(alphabet_or_token_list)
        )
        self.df.reset_index(drop=True, inplace=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> torch.Tensor:
        row = self.df.loc[index]
        tensor_h = token_string_to_tensor(row.HeavyAA_aligned, self.alphabet)
        tensor_l = token_string_to_tensor(row.LightAA_aligned, self.alphabet)
        w1, w2 = row.w1, row.w2
        condition = torch.tensor([w1, w2]).float()
        label = torch.tensor(row.single_value).float()

        return torch.cat([tensor_h, tensor_l]), condition, label
    
@dataclass
class MNISTDataset(Dataset):
    data: np.ndarray
    alphabet_or_token_list: InitVar[LabelEncoder | list[str]] = ALPHABET_AHO
    alphabet: LabelEncoder = field(init=False)

    def __post_init__(self, alphabet_or_token_list: LabelEncoder | list[str]):
        self.alphabet = (
            alphabet_or_token_list
            if isinstance(alphabet_or_token_list, LabelEncoder)
            else LabelEncoder().fit(alphabet_or_token_list)
        )


    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        row = self.data[index]
        tensor = torch.from_numpy(row).long()
        assert(type(tensor) == torch.Tensor)
        return tensor

@dataclass
class MNISTWithLabelsDataset(Dataset):
    data: np.ndarray
    labels: np.ndarray
    alphabet_or_token_list: InitVar[LabelEncoder | list[str]] = ALPHABET_AHO
    alphabet: LabelEncoder = field(init=False)

    def __post_init__(self, alphabet_or_token_list: LabelEncoder | list[str]):
        self.alphabet = (
            alphabet_or_token_list
            if isinstance(alphabet_or_token_list, LabelEncoder)
            else LabelEncoder().fit(alphabet_or_token_list)
        )


    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        row = self.data[index].reshape(-1)
        label = self.labels[index]
        tensor_row = torch.from_numpy(row).long()
        tensor_label = torch.tensor(label).long()
        assert(type(tensor_row) == torch.Tensor)
        return tensor_row, tensor_label
