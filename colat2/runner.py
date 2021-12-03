import os

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

from colat2.evaluator import Evaluator
from colat2.projectors import Projector
from colat2.trainer import Trainer
from colat2.visualizer import Visualizer

import copy


def train(cfg: DictConfig) -> None:
    """Trains model from config

    Args:
        cfg: Hydra config

    """
    # Device
    device = get_device(cfg)

    
    # Model
    # Use Hydra's instantiation to initialize directly from the config file
    model: torch.nn.Module = instantiate(cfg.model, k=cfg.k, batch_k=cfg.batch_k).to(device)
    model2: torch.nn.Module = copy.deepcopy(model) #instantiate(cfg.model, k=cfg.k).to(device)
    loss_fn: torch.nn.Module = instantiate(cfg.loss, k=min(cfg.batch_k, cfg.k)).to(device)
    generator: torch.nn.Module = instantiate(cfg.generator).to(device)
    generator2: torch.nn.Module = instantiate(cfg.generator2).to(device)
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
        model2=model2,
        loss_fn=loss_fn,
        optimizer=optimizer,
        generator=generator,
        generator2=generator2,
        projector=projector,
        batch_size=cfg.hparams.batch_size,
        iterations=cfg.hparams.iterations,
        overlap_k=cfg.hparams.overlap_k,
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
    )

    # Launch training process
    trainer.train()


def evaluate(cfg: DictConfig) -> None:
    """Evaluates model from config

    Args:
        cfg: Hydra config
    """
    # Device
    device = get_device(cfg)

    size1 = cfg.model.size if not cfg.generator.use_w else cfg.model.size * cfg.generator.num_ws
    size2 = cfg.model2.size if not cfg.generator2.use_w else cfg.model2.size * cfg.generator2.num_ws
    
    # Model
    model: torch.nn.Module = instantiate(cfg.model, k=cfg.k).to(device)
    loss_fn: torch.nn.Module = instantiate(cfg.loss).to(device)
    generator: torch.nn.Module = instantiate(cfg.generator).to(device)
    projector: torch.nn.Module = instantiate(cfg.projector).to(device)

    # Preload model
    checkpoint_path = hydra.utils.to_absolute_path(cfg.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    projector.load_state_dict(checkpoint["projector"])

    evaluator = Evaluator(
        model=model,
        loss_fn=loss_fn,
        generator=generator,
        projector=projector,
        device=device,
        batch_size=cfg.hparams.batch_size,
        iterations=cfg.hparams.iterations,
        feed_layers=cfg.feed_layers,
    )
    evaluator.evaluate()


def generate(cfg: DictConfig) -> None:
    """Generates images from config

    Args:
        cfg: Hydra config
    """
    # Device
    device = get_device(cfg)
    cfg.n_dirs = list(range(cfg.k))
    
    # Model
    # Use Hydra's instantiation to initialize directly from the config file
    model: torch.nn.Module = instantiate(cfg.model, k=cfg.k, batch_k=cfg.batch_k).to(device)
    model2: torch.nn.Module = copy.deepcopy(model) #instantiate(cfg.model, k=cfg.k).to(device)
    generator: torch.nn.Module = instantiate(cfg.generator).to(device)
    generator2: torch.nn.Module = instantiate(cfg.generator2).to(device)
    projector: torch.nn.Module = instantiate(cfg.projector).to(device)

    # Preload model
    checkpoint_path = hydra.utils.to_absolute_path(cfg.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model2.load_state_dict(checkpoint["model2"])
    projector.load_state_dict(checkpoint["projector"])

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
        gen_ind=1,
    )
    visualizer.visualize()


    visualizer2 = Visualizer(
        model=model2,
        generator=generator2,
        projector=projector,
        device=device,
        n_samples=cfg.n_samples,
        n_dirs=cfg.n_dirs,
        alphas=cfg.alphas,
        iterative=cfg.iterative,
        feed_layers=cfg.feed_layers,
        image_size=cfg.image_size,
        gen_ind=2,
    )
    visualizer2.visualize()


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
