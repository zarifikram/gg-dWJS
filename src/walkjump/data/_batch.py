from dataclasses import dataclass
from functools import cached_property

import torch

from walkjump.constants import TOKENS_AHO, TOKENS_AMP
from walkjump.utils import isotropic_gaussian_noise_like

@dataclass
class AbBatch:
    batch_tensor: torch.Tensor
    """(b, L)-shaped tensor of sequences"""
    vocab_size: int = len(TOKENS_AHO)

    @classmethod
    def from_tensor_pylist(
        cls, inputs: list[torch.Tensor], vocab_size: int = len(TOKENS_AHO)
    ) -> "AbBatch":

        packed_batch = torch.stack(inputs, dim=0)
        return cls(packed_batch, vocab_size=vocab_size)

    @cached_property
    def x(self) -> torch.Tensor:
        return torch.nn.functional.one_hot(self.batch_tensor, num_classes=self.vocab_size).float()

@dataclass
class AbWithLabelBatch:
    # TO-DO
    batch_tensor: torch.Tensor
    batch_labels: torch.Tensor
    """(b, L)-shaped tensor of sequences"""
    vocab_size: int = len(TOKENS_AHO)

    @classmethod
    def from_tensor_pylist(
        cls, inputs: list[torch.Tensor], vocab_size: int = len(TOKENS_AHO)
    ) -> "AbWithLabelBatch":
        data = []
        labels = []
        for x, y in inputs:
            data.append(x)
            labels.append(y)
        packed_batch = torch.stack(data, dim=0)
        packed_labels = torch.stack(labels, dim=0)
        return cls(packed_batch, packed_labels, vocab_size=vocab_size)

    @cached_property
    def x(self) -> torch.Tensor:
        one_hot_x = torch.nn.functional.one_hot(self.batch_tensor, num_classes=self.vocab_size).float()
        return one_hot_x + isotropic_gaussian_noise_like(one_hot_x, 1)

    @cached_property
    def y(self) -> torch.Tensor:
        return self.batch_labels.float()
    
@dataclass
class PCAbWithLabelBatch:
    # TO-DO
    batch_tensor: torch.Tensor
    batch_preferences: torch.Tensor
    batch_labels: torch.Tensor
    """(b, L)-shaped tensor of sequences"""
    vocab_size: int = len(TOKENS_AHO)

    @classmethod
    def from_tensor_pylist(
        cls, inputs: list[torch.Tensor], vocab_size: int = len(TOKENS_AHO)
    ) -> "PCAbWithLabelBatch":
        data = []
        conditions = []
        labels = []
        for x, c, y in inputs:
            data.append(x)
            conditions.append(c)
            labels.append(y)
        packed_batch = torch.stack(data, dim=0)
        packed_conditions = torch.stack(conditions, dim=0)
        packed_labels = torch.stack(labels, dim=0)
        return cls(packed_batch, packed_conditions, packed_labels, vocab_size=vocab_size)

    @cached_property
    def x(self) -> torch.Tensor:
        one_hot_x = torch.nn.functional.one_hot(self.batch_tensor, num_classes=self.vocab_size).float()
        return one_hot_x + isotropic_gaussian_noise_like(one_hot_x, 1)

    @cached_property
    def c(self) -> torch.Tensor:
        return self.batch_preferences.float()
    
    @cached_property
    def y(self) -> torch.Tensor:
        return self.batch_labels.float()
    

@dataclass
class MNISTBatch:
    batch_tensor: torch.Tensor
    """(b, L)-shaped tensor of sequences"""
    vocab_size: int = 2
    

    @classmethod
    def from_tensor_pylist(
        cls, inputs: list[torch.Tensor], vocab_size: int = 2
    ) -> "MNISTBatch":

        packed_batch = torch.stack(inputs, dim=0)
        return cls(packed_batch, vocab_size=vocab_size)

    @cached_property
    def x(self) -> torch.Tensor:
        return torch.nn.functional.one_hot(self.batch_tensor, num_classes=2).float()

@dataclass
class MNISTWithLabelsBatch:
    batch_tensor: torch.Tensor
    batch_labels: torch.Tensor
    """(b, L)-shaped tensor of sequences"""
    vocab_size: int = 2
    

    @classmethod
    def from_tensor_pylist(
        cls, inputs: list[tuple[torch.Tensor, torch.Tensor]], vocab_size: int = 2
    ) -> "MNISTWithLabelsBatch":
        data = []
        labels = []
        for x, y in inputs:
            data.append(x)
            labels.append(y)
        packed_batch = torch.stack(data, dim=0)
        packed_labels = torch.stack(labels, dim=0)
        return cls(packed_batch, packed_labels, vocab_size=vocab_size)

    @cached_property
    def x(self) -> torch.Tensor:
        return torch.nn.functional.one_hot(self.batch_tensor, num_classes=2).float() \
            + isotropic_gaussian_noise_like(torch.nn.functional.one_hot(self.batch_tensor, num_classes=2).float(), 1)
    
    @cached_property
    def y(self) -> torch.Tensor:
        return self.batch_labels.long()
    
@dataclass
class HERWithLabelsBatch:
    batch_tensor: torch.Tensor
    batch_labels: torch.Tensor
    """(b, L)-shaped tensor of sequences"""
    vocab_size: int = len(TOKENS_AHO)

    @classmethod
    def from_tensor_pylist(
        cls, inputs: list[torch.Tensor], vocab_size: int = len(TOKENS_AHO)
    ) -> "PCAbWithLabelBatch":
        data = []
        labels = []
        for x, y in inputs:
            data.append(x)
            labels.append(y)
        packed_batch = torch.stack(data, dim=0)
        packed_labels = torch.stack(labels, dim=0)
        return cls(packed_batch, packed_labels, vocab_size=vocab_size)

    @cached_property
    def x(self) -> torch.Tensor:
        one_hot_x = torch.nn.functional.one_hot(self.batch_tensor, num_classes=self.vocab_size).float()
        # return one_hot_x + isotropic_gaussian_noise_like(one_hot_x, 1) for noised discriminiator
        return one_hot_x 
    
    @cached_property
    def y(self) -> torch.Tensor:
        return self.batch_labels.float()
    
@dataclass
class AMPBatch:
    batch_tensor: torch.Tensor
    """(b, L)-shaped tensor of sequences"""
    vocab_size: int = len(TOKENS_AMP)

    @classmethod
    def from_tensor_pylist(
        cls, inputs: list[torch.Tensor], vocab_size: int = len(TOKENS_AMP)
    ) -> "AMPBatch":

        packed_batch = torch.stack(inputs, dim=0)
        return cls(packed_batch, vocab_size=vocab_size)

    @cached_property
    def x(self) -> torch.Tensor:
        one_hot_x = torch.nn.functional.one_hot(self.batch_tensor, num_classes=self.vocab_size).float()
        # return one_hot_x + isotropic_gaussian_noise_like(one_hot_x, 1) for noised discriminiator
        return one_hot_x 

@dataclass
class AMPWithLabelsBatch:
    batch_tensor: torch.Tensor
    batch_labels: torch.Tensor
    """(b, L)-shaped tensor of sequences"""
    vocab_size: int = len(TOKENS_AMP)

    @classmethod
    def from_tensor_pylist(
        cls, inputs: list[torch.Tensor], vocab_size: int = len(TOKENS_AMP)
    ) -> "AMPWithLabelsBatch":
        data = []
        labels = []
        for x, y in inputs:
            data.append(x)
            labels.append(y)
        packed_batch = torch.stack(data, dim=0)
        packed_labels = torch.stack(labels, dim=0)
        return cls(packed_batch, packed_labels, vocab_size=vocab_size)

    @cached_property
    def x(self) -> torch.Tensor:
        one_hot_x = torch.nn.functional.one_hot(self.batch_tensor, num_classes=self.vocab_size).float()
        # return one_hot_x + isotropic_gaussian_noise_like(one_hot_x, 1) for noised discriminiator
        return one_hot_x 
    
    @cached_property
    def y(self) -> torch.Tensor:
        return self.batch_labels.float()