import math
from typing import Optional

import torch
from torch.autograd import Variable
from tqdm import trange

from walkjump.model import TrainableScoreModel
import matplotlib.pyplot as plt

_DEFAULT_SAMPLING_OPTIONS = {"delta": 0.5, "friction": 1.0, "lipschitz": 1.0, "steps": 100}


def sachsetal(
    model: TrainableScoreModel,
    y: torch.Tensor,
    v: torch.Tensor,
    sampling_options: dict[str, float | int] = _DEFAULT_SAMPLING_OPTIONS,
    mask_idxs: Optional[list[int]] = None,
    save_trajectory: bool = False,
    verbose: bool = True,
    guidance: bool = False,
    label: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:

    options = _DEFAULT_SAMPLING_OPTIONS | sampling_options  # overwrite

    delta, gamma, lipschitz = options["delta"], options["friction"], options["lipschitz"]

    step_iterator = (
        trange(int(options["steps"]), desc="Sachs, et al", leave=False)
        if verbose
        else range(int(options["steps"]))
    )
    # we will plot 10 samples for each 4 steps.
    fig, axes = plt.subplots(int(options["steps"] / 4), 10, figsize=(15, 10))

    with torch.set_grad_enabled(model.needs_gradients):
        u = pow(lipschitz, -1)  # inverse mass
        zeta1 = math.exp(
            -gamma
        )  # gamma is effective friction here (originally 'zeta1 = math.exp(-gamma * delta)')
        zeta2 = math.exp(-2 * gamma)

        traj = []
        for _i in step_iterator:
            # y += delta * v / 2  # y_{t+1}
            y = y + delta * v / 2

            # if _i % 4 == 0:
            #     for i, ax in enumerate(axes[_i//4]):
            #         arr = y[i]
            #         ax.imshow(arr.argmax(-1).reshape(28, 28).cpu(), cmap="gray")
            #         ax.axis("off")
            psi = model(y)
            if guidance:
                guide = get_label_gradient_wrt_x(model.guide_model, y, label) / pow(model.sigma, 2)
                psi = psi + guide

                # if guide_model2 exists, use it as well
                if hasattr(model, "guide_model2"):
                    guide2 = get_label_gradient_wrt_x(model.guide_model2, y, label) / pow(model.sigma, 2)
                    psi = psi - guide2

            noise_update = torch.randn_like(y)
            # prevent updates in masked positions
            if mask_idxs and False: #TODO: FIX this
                psi[:, mask_idxs, :] = 0.0
                noise_update[:, mask_idxs, :] = 0.0
            v += u * delta * psi / 2  # v_{t+1}
            v = (
                zeta1 * v + u * delta * psi / 2 + math.sqrt(u * (1 - zeta2)) * noise_update
            )  # v_{t+1}
            # y += delta * v / 2  # y_{t+1}
            y = y + delta * v / 2
            # gc.collect()
            # torch.cuda.empty_cache()
            if save_trajectory:
                traj.append(y)
        # plt.savefig(f"figs/mnist_each_steps.png")
        return y, v, traj

def get_label_gradient_wrt_x(guide_model, x, y):
    """
    Get the gradient of the label wrt x. Specifically, \delta_{x}log(P(Y=y|x))
    """
    # first, clear the gradient zero the gradient
    # model.guide_model.model.zero_grad()
    torch.set_grad_enabled(True)
    # print(f"X shape : {x.shape}")
    x = Variable(x, requires_grad=True)
    log_p_y = guide_model.model(x)
    log_p_y = log_p_y[:, y] # select the log probability of the correct label
    ones = torch.ones_like(log_p_y)

    # Backward pass starting from the selected probabilities
    log_p_y.backward(gradient=ones)

    # Access the gradients of the input 
    grad = x.grad
    # print(f"Grad shape: {grad.shape}")
    torch.set_grad_enabled(False)
    return grad