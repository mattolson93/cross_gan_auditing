import os

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

from colat.evaluator import Evaluator
from colat.projectors import Projector
from colat.trainer import Trainer
from colat.visualizer import Visualizer


def train(cfg: DictConfig) -> None:
    """Trains model from config

    Args:
        cfg: Hydra config

    """
    # Device
    device = get_device(cfg)

    #import pdb; pdb.set_trace()
    # Model
    # Use Hydra's instantiation to initialize directly from the config file
    #size = cfg.model.size if not cfg.generator.use_w else cfg.model.size * cfg.generator.num_ws
    model: torch.nn.Module = instantiate(cfg.model, k=cfg.k, batch_k=cfg.batch_k).to(device)
    loss_fn: torch.nn.Module = instantiate(cfg.loss, k=cfg.k).to(device)
    generator: torch.nn.Module = instantiate(cfg.generator).to(device)
    projector: Projector = instantiate(cfg.projector).to(device)

    optimizer: torch.optim.Optimizer = instantiate(
        cfg.hparams.optimizer,
        list(model.parameters()) + list(projector.parameters())
        if cfg.train_projector
        else model.parameters(),
    )
    scheduler = instantiate(cfg.hparams.scheduler, optimizer)

    # Paths
    save_path = os.getcwd() if cfg.save else None
    checkpoint_path = (
        hydra.utils.to_absolute_path(cfg.checkpoint)
        if cfg.checkpoint is not None
        else None
    )

    # Tensorboard
    if cfg.tensorboard:
        # Note: global step is in epochs here
        writer = SummaryWriter(os.getcwd())
        # Indicate to TensorBoard that the text is pre-formatted
        text = f"<pre>{OmegaConf.to_yaml(cfg)}</pre>"
        writer.add_text("config", text)
    else:
        writer = None

    # Trainer init
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        generator=generator,
        projector=projector,
        batch_size=cfg.hparams.batch_size,
        iterations=cfg.hparams.iterations,
        device=device,
        eval_freq=cfg.eval_freq,
        eval_iters=cfg.eval_iters,
        scheduler=scheduler,
        grad_clip_max_norm=cfg.hparams.grad_clip_max_norm,
        writer=writer,
        save_path=save_path,
        checkpoint_path=checkpoint_path,
        mixed_precision=cfg.mixed_precision,
        train_projector=cfg.train_projector,
        feed_layers=cfg.feed_layers,
        dre_path=cfg.dre_path,

    )

    # Launch training process
    trainer.train()

    cfg.n_dirs = list(range(cfg.k))
    visualizer = Visualizer(
        model=model,
        generator=generator,
        projector=projector,
        device=device,
        n_samples=cfg.n_samples,
        n_dirs=cfg.n_dirs,
        alphas=cfg.alphas,
        iterative=cfg.iterative,
        feed_layers=cfg.feed_layers,
        image_size=cfg.image_size,
    )
    visualizer.visualize()

    evaluator = Evaluator(
        model=model,
        generator=generator,
        device=device,
        batch_size=cfg.hparams.batch_size,
        iterations=cfg.hparams.iterations,
        total_directions=cfg.k,
        att_model_path=os.path.join(hydra.utils.get_original_cwd(),"att_classifier.pt")
    )
    score = evaluator.evaluate()

    with open('entropy.txt', 'w') as f:       f.write(f"{score}")
    with open(f"entropy_{score}", 'w') as f:  f.write("")

    trunc_val = 1.0
    score = evaluator.evaluate(trunc_val)

    with open(f'entropy_t{trunc_val}.txt', 'w') as f:       f.write(f"{score}")
    with open(f"entropy_t{trunc_val}_s{score}", 'w') as f:  f.write("")


def evaluate(cfg: DictConfig) -> None:
    """Evaluates model from config

    Args:
        cfg: Hydra config
    """
    # Device
    device = get_device(cfg)

    # Model
    model: torch.nn.Module = instantiate(cfg.model, k=cfg.k, batch_k=cfg.batch_k).to(device)
    #loss_fn: torch.nn.Module = instantiate(cfg.loss).to(device)
    generator: torch.nn.Module = instantiate(cfg.generator).to(device)
    #projector: torch.nn.Module = instantiate(cfg.projector).to(device)

    # Preload model
    checkpoint = torch.load(cfg.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    #projector.load_state_dict(checkpoint["projector"])

    evaluator = Evaluator(
        model=model,
        generator=generator,
        device=device,
        batch_size=cfg.hparams.batch_size,
        iterations=cfg.hparams.iterations,
        total_directions=cfg.k,
        att_model_path=os.path.join(hydra.utils.get_original_cwd(),"att_classifier.pt")
    )
    score = evaluator.evaluate()

    with open('diveristy.txt', 'w') as f:       f.write(f"{score}")
    with open(f"diveristy_{score}", 'w') as f:  f.write("")

    trunc_val = 1.0
    #score = evaluator.evaluate(trunc_val)

    #with open(f'entropy_t{trunc_val}.txt', 'w') as f:       f.write(f"{score}")
    #with open(f"entropy_t{trunc_val}_s{score}", 'w') as f:  f.write("")




def generate(cfg: DictConfig) -> None:
    """Generates images from config

    Args:
        cfg: Hydra config
    """
    # Device
    device = get_device(cfg)
    #import pdb; pdb.set_trace()
    cfg.n_dirs = list(range(cfg.k))

    # Model
    model: torch.nn.Module = instantiate(cfg.model, k=cfg.k, batch_k=cfg.batch_k).to(device)
    generator: torch.nn.Module = instantiate(cfg.generator).to(device)
    projector: torch.nn.Module = instantiate(cfg.projector).to(device)

    # Preload model
    checkpoint = torch.load(cfg.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    #projector.load_state_dict(checkpoint["projector"])

    #import pdb; pdb.set_trace()
    #from sklearn.metrics.pairwise import cosine_similarity as cos
    #cos(model.state_dict()['params'].cpu().numpy())

    visualizer = Visualizer(
        model=model,
        generator=generator,
        device=device,
        n_samples=cfg.n_samples,
        n_dirs=cfg.n_dirs,
        alphas=cfg.alphas,
        iterative=cfg.iterative,
        feed_layers=cfg.feed_layers,
        image_size=cfg.image_size,
    )
    visualizer.visualize()
    visualizer.visualize(.7)


def get_device(cfg: DictConfig) -> torch.device:
    """Initializes the device from config

    Args:
        cfg: Hydra config

    Returns:
        device on which the model will be trained or evaluated

    """
    if cfg.auto_cpu_if_no_gpu:
        device = (
            torch.device(cfg.device)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    else:
        device = torch.device(cfg.device)

    return device
