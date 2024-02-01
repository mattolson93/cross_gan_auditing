import hydra
from omegaconf import DictConfig
import omegaconf
import os

import colat2.runner as runner
import colat2.utils.log_utils as utils


@hydra.main(config_path="conf", config_name="eval2")
def eval(cfg: DictConfig):
    #import pdb; pdb.set_trace()
    mycfg2 = omegaconf.OmegaConf.load(os.path.join(hydra.utils.get_original_cwd(), cfg.cfg2, ".hydra/config.yaml"))
    utils.display_config(cfg)
    utils.display_config(mycfg2)
    runner.evaluate_two_seperates(cfg, mycfg2)


if __name__ == "__main__":
    eval()
