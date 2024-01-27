import torch
from torch import nn

from walkjump.data import AbWithLabelBatch
from walkjump.utils import isotropic_gaussian_noise_like

from ._base import DiscriminatorModel

class AbDiscriminatorModel(DiscriminatorModel):
    needs_gradients: bool = False

    def compute_loss(self, batch: AbWithLabelBatch) -> torch.Tensor:
        y = batch.y
        xhat = self(batch.x).squeeze(-1)

        # return mse loss
        return nn.functional.mse_loss(xhat, y).mean()
