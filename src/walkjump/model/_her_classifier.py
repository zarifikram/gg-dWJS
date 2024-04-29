import torch
from torch import nn

from walkjump.data import HERWithLabelsBatch
from walkjump.utils import isotropic_gaussian_noise_like

from ._base import ClassifierModel


class HERClassifierModel(ClassifierModel):
    needs_gradients: bool = False

    def compute_loss(self, batch: HERWithLabelsBatch) -> torch.Tensor:
        y = batch.y # true label yes/no
        xhat = self.model(batch.x) # tensor after sigmoid
        
        # now we do binary cross entropy
        return nn.functional.binary_cross_entropy(xhat, y)
