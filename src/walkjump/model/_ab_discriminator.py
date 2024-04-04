import torch
from torch import nn

from walkjump.data import AbWithLabelBatch, PCAbWithLabelBatch
from walkjump.utils import isotropic_gaussian_noise_like

from ._base import DiscriminatorModel, PreferenceConditionalDiscriminatorModel

class AbDiscriminatorModel(DiscriminatorModel):
    needs_gradients: bool = False

    def compute_loss(self, batch: AbWithLabelBatch) -> torch.Tensor:
        y = batch.y
        xhat = self(batch.x).squeeze(-1)

        # return mse loss
        return nn.functional.mse_loss(xhat, y).mean()
    
class PCAbDiscriminatorModel(PreferenceConditionalDiscriminatorModel):
    needs_gradients: bool = False

    def compute_loss(self, batch: PCAbWithLabelBatch) -> torch.Tensor:
        y = batch.y
        xhat = self(batch.x, batch.c).squeeze(-1)

        # return mse loss
        return nn.functional.mse_loss(xhat, y).mean()
