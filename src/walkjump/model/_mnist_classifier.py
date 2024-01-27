import torch
from torch import nn

from walkjump.data import MNISTWithLabelsBatch
from walkjump.utils import isotropic_gaussian_noise_like

from ._base import ClassifierModel


class MNISTClassifierModel(ClassifierModel):
    needs_gradients: bool = False

    def compute_loss(self, batch: MNISTWithLabelsBatch) -> torch.Tensor:
        y = batch.y
        xhat = self.model(batch.x)

        return nn.NLLLoss()(xhat, y)
