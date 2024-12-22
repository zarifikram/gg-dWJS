import hydra
import torch
from walkjump.constants._tokens import ALPHABET_AMP
import wandb
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf

from walkjump.cmdline.utils import instantiate_redesign_mask, instantiate_seeds
from walkjump.sampling import walkjump, walkjump_static_mnist


@hydra.main(version_base=None, config_path="../hydra_config", config_name="sample_amp")
def sample_amp(cfg: DictConfig) -> bool:
    log_cfg = OmegaConf.to_container(cfg, throw_on_missing=True, resolve=True)

    wandb.require("service")
    if rank_zero_only.rank == 0:
        print(OmegaConf.to_yaml(log_cfg))

    hydra.utils.instantiate(cfg.setup)

    if cfg.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"using device {device}")
    else:
        print(f"using device {cfg.device}")
        device = torch.device(cfg.device)

    mask_idxs = instantiate_redesign_mask(cfg.designs.redesign_regions or [])
    
    print(f"Redesigning {len(mask_idxs)} positions")
    seeds = instantiate_seeds(cfg.designs)
    print(f"Number of seeds {len(seeds)}")
    
    if not cfg.dryrun:
        model = hydra.utils.instantiate(cfg.model).to(device)
        sample_df = walkjump(
            seeds,
            model,
            delta=cfg.langevin.delta,
            lipschitz=cfg.langevin.lipschitz,
            friction=cfg.langevin.friction,
            steps=cfg.langevin.steps,
            num_samples=cfg.designs.num_samples,
            mask_idxs=mask_idxs,
            chunksize=cfg.designs.chunksize,
            guidance = cfg.guidance,
            alphabet = ALPHABET_AMP,
        )
        # sample_df.drop_duplicates(subset=["fv_heavy_aho", "fv_light_aho"], inplace=True)
        print(f"Writing {len(sample_df)} samples to {cfg.designs.output_csv}")
        sample_df.to_csv(cfg.designs.output_csv, index=False)
    
    return True
