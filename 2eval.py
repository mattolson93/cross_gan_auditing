import hydra
from omegaconf import DictConfig

import colat2.runner as runner
import colat2.utils.log_utils as utils


@hydra.main(config_path="conf", config_name="eval2")
def eval(cfg: DictConfig):
    utils.display_config(cfg)
    runner.evaluate(cfg)


if __name__ == "__main__":
    eval()
