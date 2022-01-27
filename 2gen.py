import hydra
from omegaconf import DictConfig

import colat2.runner as runner
import colat2.utils.log_utils as utils


@hydra.main(config_path="conf", config_name="gen2")
def gen(cfg: DictConfig):
    #import pdb; pdb.set_trace()
    utils.display_config(cfg)
    runner.generate(cfg)


if __name__ == "__main__":
    gen()
