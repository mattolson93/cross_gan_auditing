import os
import lpips

import hydra
import torch
from torch.nn import functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

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
    model2: torch.nn.Module = instantiate(cfg.model, k=cfg.k, batch_k=cfg.batch_k).to(device)
    loss_fn: torch.nn.Module = instantiate(cfg.loss, k=min(cfg.batch_k, cfg.k)).to(device)
    generator: torch.nn.Module = instantiate(cfg.generator).to(device)
    generator2: torch.nn.Module = instantiate(cfg.generator2).to(device)
    projector: Projector = instantiate(cfg.projector).to(device)

    optimizer: torch.optim.Optimizer = instantiate(
        cfg.hparams.optimizer,
        list(model.parameters()) + list(model2.parameters()) + list(projector.parameters())
        if cfg.train_projector
        else list(model.parameters()) + list(model2.parameters()),
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
        overlap_k=cfg.overlap_k,
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

    cfg.n_dirs = list(range(cfg.k))
    #import pdb; pdb.set_trace()
    invert_z_file = os.path.join( hydra.utils.get_original_cwd(), 'invertedpoints',cfg['generator']['class_name']+ "_" +cfg['generator2']['class_name'] + ".pt")

    if os.path.exists(invert_z_file):
        zs = torch.load(invert_z_file, map_location=device)
        z1 = zs[:cfg.n_samples,0]
        z2 = zs[:cfg.n_samples,1]
    
        if generator.w_primary:  z1 = generator.convert_z2w(z1)
        if generator2.w_primary: z2 = generator2.convert_z2w(z2)

    else:
        z1, z2 = invert_2_gens(generator, generator2, cfg.n_samples)
        torch.save(torch.stack([z1,z2],dim=1),invert_z_file)
    #from sklearn.metrics.pairwise import cosine_similarity as cos
    #cos(model.state_dict()['params'].cpu().numpy())

    visualizer = Visualizer(
        model=model,
        generator=generator,
        projector=projector,
        device=device,
        #n_samples=cfg.n_samples,
        n_samples=z1,
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
        #n_samples=cfg.n_samples,
        n_samples=z2,
        n_dirs=cfg.n_dirs,
        alphas=cfg.alphas,
        iterative=cfg.iterative,
        feed_layers=cfg.feed_layers,
        image_size=cfg.image_size,
        gen_ind=2,
    )
    visualizer2.visualize()

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
    checkpoint = torch.load(cfg.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model2.load_state_dict(checkpoint["model2"])
    projector.load_state_dict(checkpoint["projector"])


    '''zz = model.params.detach().cpu().numpy()
    z1 = zz[2]
    z2 = zz[1]

    steps = z1.shape[0]
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity as cos
    from scipy import spatial

    for i in range(steps):
        best_ind = 0
        best_score = 0
        for j in range(z1.shape[0]):
            cur_z1 = z1[np.arange(len(z1))!=j].reshape(1,-1)
            cur_z2 = z2[np.arange(len(z2))!=j].reshape(1,-1)

            cur_score = 1 - spatial.distance.cosine(cur_z1, cur_z2)
            if cur_score > best_score:
                best_score = cur_score
                best_ind = j

        z1 = np.delete(z1, best_ind)
        z2 = np.delete(z2, best_ind)
        print(i, " ", 1 - spatial.distance.cosine(z1,z2))

    '''


    invert_z_file = os.path.join( hydra.utils.get_original_cwd(), 'invertedpoints',cfg['generator']['class_name']+ "_" +cfg['generator2']['class_name'] + ".pt")

    if os.path.exists(invert_z_file):
        zs = torch.load(invert_z_file, map_location=device)
        z1 = zs[:cfg.n_samples,0]
        z2 = zs[:cfg.n_samples,1]
    
        if generator.w_primary:  z1 = generator.convert_z2w(z1)
        if generator2.w_primary: z2 = generator2.convert_z2w(z2)

    else:
        z1, z2 = invert_2_gens(generator, generator2, cfg.n_samples)
        torch.save(torch.stack([z1,z2],dim=1),invert_z_file)
   

    visualizer = Visualizer(
        model=model,
        generator=generator,
        projector=projector,
        device=device,
        #n_samples=cfg.n_samples,
        n_samples=z1,
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
        #n_samples=cfg.n_samples,
        n_samples=z2,
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

def invert_2_gens(G1, G2, n_samples):
    convert2w_1 = G1.w_primary
    convert2w_2 = G2.w_primary
    
    G1.use_z()
    G2.use_z()

    #get 2 semi-random starting point images
    z1 = G1.sample_latent(n_samples).detach() / 10
    z2 = G2.sample_latent(n_samples).detach() / 10

    z1.requires_grad = True
    z2.requires_grad = True


    optimizer = torch.optim.Adam([z1, z2], lr=1e-3)

    pbar = tqdm(range(500))
    latent_path = []
    percept = lpips.LPIPS(net='vgg', spatial=True).to(G1.device)

    l2_lam = .01
    mse_lam = 1

    for i in pbar:

        img1 = G1(z1)
        img2 = G2(z2)

        mse_loss = mse_lam*F.mse_loss(img1, img2)
        l2_loss  = l2_lam *(z1 **2  + z2 **2).mean()
        p_loss = 20*percept(img1, img2).mean()

        loss =  mse_loss + l2_loss + p_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_description( f' mse: {mse_loss.item():.4f}; gaussian: {l2_loss.item():.4f}')
    #
    if convert2w_1: 
        z1 = G1.convert_z2w(z1)
        G1.use_w()
    if convert2w_2: 
        z2 = G2.convert_z2w(z2)
        G2.use_w()

    print(z1.max())
    print(z2.max())
    print(z1.min())
    print(z2.min())
    return z1.detach(), z2.detach(), img1, img2