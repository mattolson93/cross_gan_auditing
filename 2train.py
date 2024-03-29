import hydra
from omegaconf import DictConfig

import colat2.runner as runner
import colat2.utils.log_utils as utils


@hydra.main(config_path="conf", config_name="train2")
def train(cfg: DictConfig):
    
    utils.display_config(cfg)
    runner.train(cfg)


if __name__ == "__main__":
    train()
