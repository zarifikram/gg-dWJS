from typing import Callable, Optional

import pandas as pd
import torch
from torch.autograd import Variable

from walkjump.constants import ALPHABET_AHO, LENGTH_FV_HEAVY_AHO, LENGTH_FV_LIGHT_AHO
from walkjump.model import TrainableScoreModel
from walkjump.utils import token_string_from_tensor, token_string_to_tensor

from ._sampler_factory import create_sampler_fn


def stack_seed_sequences(seeds: list[str], num_samples: int) -> torch.Tensor:
    """
    Convert a list of seed sequences to to a collection of initial points for sampling trajectory.

    Parameters
    ----------
    seeds: List[str]
        List of input seeeds
    num_samples: int
        Number of samples per seed to generate

    Returns
    -------
    torch.Tensor
        Padded seed tensor of shape (len(seeds) * num_samples, max(map(len, seeds)), VOCAB_SIZE)
    """
    return torch.nn.functional.one_hot(
        torch.nn.utils.rnn.pad_sequence(
            [token_string_to_tensor(seed_i, ALPHABET_AHO, onehot=False) for seed_i in seeds],
            batch_first=True,
        ).repeat_interleave(num_samples, dim=0),
        num_classes=len(ALPHABET_AHO.classes_),
    ).float()
    

def stack_seed_sequences_mnist(seeds: list[str], num_samples: int) -> torch.Tensor:
    """
    Convert a list of seed sequences to to a collection of initial points for sampling trajectory.

    Parameters
    ----------
    seeds: List[str]
        List of input seeeds
    num_samples: int
        Number of samples per seed to generate

    Returns
    -------
    torch.Tensor
        Padded seed tensor of shape (len(seeds) * num_samples, max(map(len, seeds)), VOCAB_SIZE)
    """
    return torch.nn.functional.one_hot(
        torch.nn.utils.rnn.pad_sequence(
            [token_string_to_tensor(seed_i, ALPHABET_AHO, onehot=False) for seed_i in seeds],
            batch_first=True,
        ).repeat_interleave(num_samples, dim=0),
        num_classes=2,
    ).float()

def walk(
    seed_tensor: torch.Tensor,
    model: TrainableScoreModel,
    sampler_fn: Callable,
    chunksize: int = 1,
    save_trajectory: bool = False,
    guidance: bool = False,
    label: int = 0,
) -> torch.Tensor:
    """
    Walk step

    Parameters
    ----------
    seed_tensor: torch.Tensor
        Stacked seed batch
    model: ScoreModel
        Model of Y-manifold used for walking
    sampler_fn: Callable
        The (partial) function with sampling parameters
    chunksize: int
        Used for chunking the batch to save memory. Providing
        chunksize = N will force the sampling to occur in N batches.

    Returns
    -------
    torch.Tensor
        Samples from Y
    """
    seed_tensor = seed_tensor.to(model.device)  # type: ignore[arg-type]
    list_ys = []

    for seed_chunk in seed_tensor.chunk(chunksize):

        # note: apply_noise should control whether seed_chunk.requires_grad
        seed_chunk = model.apply_noise(seed_chunk)
        
        ys, *v_trajectory = sampler_fn(
            model, seed_chunk, torch.zeros_like(seed_chunk), save_trajectory=save_trajectory, guidance=guidance, label=label
        )
        if save_trajectory:
            list_ys.extend(v_trajectory[-1])
        else:
            list_ys.append(ys.detach())

    y = torch.cat(list_ys, dim=0)

    return y


def jump(y: torch.Tensor, model: TrainableScoreModel, chunksize: int = 1, guidance: bool = False, label: int = 0) -> torch.Tensor:
    """
    Jump step. Bring samples from Y back to X manifold.

    Parameters
    ----------
    ys: torch.Tensor
        samples of Y
    model: ScoreModel
        Bayes estimator of X
    chunksize: int
        Used for chunking the batch to save memory. Providing
        chunksize = N will force the sampling to occur in N batches.

    Returns
    -------
    torch.Tensor
        Samples from X
    """
    list_xhats = []
    for y_chunk in y.chunk(chunksize):
        with torch.set_grad_enabled(model.needs_gradients):
            xhat_chunk = model.xhat(y_chunk).cpu()
            torch.cuda.empty_cache()
            if guidance:
                if hasattr(model, "guide_model"):
                    guide = get_label_gradient_wrt_x(model.guide_model, y_chunk, label) / pow(model.sigma, 2)
                    xhat_chunk = xhat_chunk + guide.cpu()
                    torch.cuda.empty_cache()
                # if guide_model2 exists, use it as well
                if hasattr(model, "guide_model2"):
                    guide2 = get_label_gradient_wrt_x(model.guide_model2, y_chunk, label) / pow(model.sigma, 2)
                    xhat_chunk = xhat_chunk - guide2.cpu()
                    torch.cuda.empty_cache()
            # print(f"torch xhat {xhat_chunk[0][0][:20]} modelY {model(y_chunk).cpu()[0][0][:20]}")

        list_xhats.append(xhat_chunk)

    xhats = torch.cat(list_xhats, dim=0)
    return xhats

def get_label_gradient_wrt_x(guide_model, x, y):
    """
    Get the gradient of the label wrt x. Specifically, \delta_{x}log(P(Y=y|x))
    """
    # first, clear the gradient zero the gradient
    # model.guide_model.model.zero_grad()
    torch.set_grad_enabled(True)
    # print(f"X shape : {x.shape}")
    x = Variable(x, requires_grad=True)
    # c = torch.tensor([0, 1]).float().to(x.device)
    # c should have the same batch size as x
    # c = c.repeat(x.shape[0], 1)

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


def walkjump(
    seed: str | list[str],
    model: TrainableScoreModel,
    delta: float = 0.5,
    lipschitz: float = 1.0,
    friction: float = 1.0,
    steps: int = 100,
    num_samples: int = 100,
    verbose: bool = True,
    mask_idxs: Optional[list[int]] = None,
    chunksize: int = 1,
    guidance: bool = False,
) -> pd.DataFrame:
    """
    Sample sequences

    See: https://arxiv.org/abs/1909.05503

    Parameters
    ----------
    seed: Union[str, List[str]]
        Either a single seed sequence or a list of sequences.
    model: DeepEnergyModel
        Model equipped with a .score() method that returns gradient of model wrt seed sequence.
    delta: float
        Step size
    lipschitz: float
        Lipschitz constant
    friction: float
        Dampening term
    steps: int
        Number of steps in chain
    num_samples: int
        Number of samples to produce
    affixes: bool
        Prepend/append affix tokens (start, stop) to input.
    mask_idxs: Optional[List[int]]
        Indices in seed str of residues to preserve during sampling

    Returns
    -------
    List[str]
        Sampled sequences
    """
    # bring the seed_tensor to the "Y manifold"
    seed_tensor = stack_seed_sequences([seed] if isinstance(seed, str) else seed, num_samples)
    # print(f"Seed shape {seed.shape}")
    print(f"Seed tensor shape {seed_tensor.shape}")
    # keep original x in masked positions
    # if mask_idxs: 
    #     seed_tensor_masked = seed_tensor[:, mask_idxs, :].clone()

    assert delta < 1

    sampler_fn = create_sampler_fn(
        verbose=verbose,
        mask_idxs=mask_idxs,
        delta=delta * model.sigma,
        friction=friction,
        lipschitz=lipschitz,
        steps=steps,
    )

    ys = walk(seed_tensor, model, sampler_fn, chunksize=chunksize, save_trajectory=False, guidance=guidance)
    xhats = jump(ys, model, chunksize=chunksize, guidance=guidance)

    # TODO: do we need to restore original x in masked positions
    if mask_idxs and False: #TODO: fix this
        xhats[:, mask_idxs, :] = seed_tensor_masked

    seqs = token_string_from_tensor(xhats, ALPHABET_AHO, from_logits=True)

    fv_heavy_aho_sample_list = [seq[:LENGTH_FV_HEAVY_AHO] for seq in seqs]
    fv_light_aho_sample_list = [seq[LENGTH_FV_HEAVY_AHO:] for seq in seqs]

    fv_heavy_aho_seed_list = token_string_from_tensor(
        seed_tensor[:, :LENGTH_FV_HEAVY_AHO], ALPHABET_AHO, from_logits=True
    )
    fv_light_aho_seed_list = token_string_from_tensor(
        seed_tensor[:, LENGTH_FV_HEAVY_AHO:], ALPHABET_AHO, from_logits=True
    )

    return pd.DataFrame(
        {
            "fv_heavy_aho": fv_heavy_aho_sample_list,
            "fv_light_aho": fv_light_aho_sample_list,
            "fv_heavy_aho_seed": fv_heavy_aho_seed_list,
            "fv_light_aho_seed": fv_light_aho_seed_list,
        }
    )

def walkjump_static_mnist(
    seed: str | list[str],
    model: TrainableScoreModel,
    delta: float = 0.5,
    lipschitz: float = 1.0,
    friction: float = 1.0,
    steps: int = 100,
    num_samples: int = 100,
    verbose: bool = True,
    mask_idxs: Optional[list[int]] = None,
    chunksize: int = 1,
    guidance: bool = False,
) -> pd.DataFrame:
    """
    Sample sequences

    See: https://arxiv.org/abs/1909.05503

    Parameters
    ----------
    seed: Union[str, List[str]]
        Either a single seed sequence or a list of sequences.
    model: DeepEnergyModel
        Model equipped with a .score() method that returns gradient of model wrt seed sequence.
    delta: float
        Step size
    lipschitz: float
        Lipschitz constant
    friction: float
        Dampening term
    steps: int
        Number of steps in chain
    num_samples: int
        Number of samples to produce
    affixes: bool
        Prepend/append affix tokens (start, stop) to input.
    mask_idxs: Optional[List[int]]
        Indices in seed str of residues to preserve during sampling

    Returns
    -------
    List[str]
        Sampled sequences
    """
    seqs_all = []
    seeds_all = []
    
    for label in range(10):
        # bring the seed_tensor to the "Y manifold"
        seed_tensor = stack_seed_sequences_mnist([seed] if isinstance(seed, str) else seed, num_samples)
        # print(f"Seed shape {seed.shape}")
        print(f"Seed tensor shape {seed_tensor.shape}")
        # keep original x in masked positions
        if mask_idxs: 
            seed_tensor_masked = seed_tensor[:, mask_idxs, :].clone()

        assert delta < 1

        sampler_fn = create_sampler_fn(
            verbose=verbose,
            mask_idxs=mask_idxs,
            delta=delta * model.sigma,
            friction=friction,
            lipschitz=lipschitz,
            steps=steps,
        )

        # print(f"Model {model.guide_model}")

        ys = walk(seed_tensor, model, sampler_fn, chunksize=chunksize, save_trajectory=False, guidance=guidance, label=label)
        xhats = jump(ys, model, chunksize=chunksize, guidance=guidance, label=label)
        # xhats = ys

        # TODO: do we need to restore original x in masked positions
        if mask_idxs and False: #TODO: fix this
            xhats[:, mask_idxs, :] = seed_tensor_masked

        seqs = token_string_from_tensor(xhats, ALPHABET_AHO, from_logits=True)
        seeds = token_string_from_tensor(seed_tensor, ALPHABET_AHO, from_logits=True)

        seqs_all.extend(seqs)
        seeds_all.extend(seeds)
    # fv_heavy_aho_sample_list = [seq[:LENGTH_FV_HEAVY_AHO] for seq in seqs]
    # fv_light_aho_sample_list = [seq[LENGTH_FV_HEAVY_AHO:] for seq in seqs]

    # fv_heavy_aho_seed_list = token_string_from_tensor(
    #     seed_tensor[:, :LENGTH_FV_HEAVY_AHO], ALPHABET_AHO, from_logits=True
    # )
    # fv_light_aho_seed_list = token_string_from_tensor(
    #     seed_tensor[:, LENGTH_FV_HEAVY_AHO:], ALPHABET_AHO, from_logits=True
    # )

    return pd.DataFrame(
        {
            "generated" : seqs_all,
            "seed" : seeds_all
        }
    )
