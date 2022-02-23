import os
import lpips

import sys
# insert at 1, 0 is the script path (or '' in REPL)
#sys.path.insert(1, '../colat')
sys.path.insert(1, '/g/g15/olson60/research/latentclr/')


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

from colat.utils.net_utils import create_dre_model
import torchvision
from torchvision.utils import save_image
from torchvision.utils import make_grid

def dre_loss(logits, batch_size):
    dre_logit = F.softplus(dclamp(logits,min=-50, max=50))

    inlier_loss = -torch.log(dre_logit[:batch_size]) 
    outlier_loss = dre_logit[batch_size:] 

    return inlier_loss.mean() + outlier_loss.mean()


def mymain(cfg: DictConfig) -> None:

    device = get_device(cfg)

    
    # Model
    # Use Hydra's instantiation to initialize directly from the config file
    generator: torch.nn.Module = instantiate(cfg.generator).to(device)
    #import pdb; pdb.set_trace()
    generator2: torch.nn.Module = instantiate(cfg.generator2).to(device)
    

    projector: Projector = instantiate(cfg.projector).to(device)
    #if False:
    #projector = AttClsModel("resnet18", cfg.projector.layers, is_pretrained=False).to(device)
    #projector.net.load_state_dict(torch.load(hydra.utils.to_absolute_path(cfg.model_path)), strict=False)
    


    dre_model = create_dre_model(layers=cfg.projector.layers) 

    optimizer: torch.optim.Optimizer = instantiate(
        cfg.hparams.optimizer,
        list(dre_model.parameters()),
    )

    batch_size = cfg.hparams.batch_size

    dre_model.train()
    projector.eval()
    generator.eval()
    generator2.eval()

    resize = torchvision.transforms.Resize(128)



    for i in tqdm(range(1000)):
        optimizer.zero_grad()
        with torch.no_grad():
            img1 = generator( generator.sample_latent(batch_size).to(device))
            img2 = generator2(generator2.sample_latent(batch_size).to(device))
            #import pdb; pdb.set_trace()
            imgs = torch.cat([resize(img1),resize(img2)],dim=0)
            
            #print(imgs.max(), imgs.min())

            feats = projector(imgs)
        #import pdb; pdb.set_trace()
        logits = dre_model(feats)
        labels = torch.cat([torch.zeros(batch_size), torch.ones(batch_size)],dim=0)
        #labels = F.one_hot(labels.long(),2).float().cuda()

        loss = dre_loss(logits,batch_size)
        if i % 10 == 0:
            acc1 = (logits[:batch_size ]).mean().item()
            acc2 = (logits[batch_size: ]).mean().item()
            acc = (acc2 + acc1)/(batch_size*2)
            print(f"acc1: {acc1:.3f}, acc2: {acc2:.3f}, loss {loss}")

            #import pdb; pdb.set_trace()
    


        loss.backward()
        optimizer.step()
        
    torch.save(dre_model.state_dict(),"dre.pt")
    _ , new_inds = torch.sort(logits[:batch_size].squeeze())
    out_imgs = make_grid(img1.clip(0,1)[new_inds]).cpu().detach()
    save_image(out_imgs,f"{i:05d}_{acc:.3f}.png")



    print("done")


def oldmain(cfg: DictConfig) -> None:

    device = get_device(cfg)

    
    # Model
    # Use Hydra's instantiation to initialize directly from the config file
    generator: torch.nn.Module = instantiate(cfg.generator).to(device)
    generator2: torch.nn.Module = instantiate(cfg.generator2).to(device)
    projector: Projector = instantiate(cfg.projector).to(device)

    dre_model = create_dre_model(layers=cfg.projector.layers) 

    optimizer: torch.optim.Optimizer = instantiate(
        cfg.hparams.optimizer,
        list(dre_model.parameters()),
    )

    batch_size = cfg.hparams.batch_size

    dre_model.train()
    projector.eval()
    generator.eval()
    generator2.eval()

    bce_loss = torch.nn.BCEWithLogitsLoss()

    for i in tqdm(range(250)):
        optimizer.zero_grad()
        with torch.no_grad():
            img1 = generator( generator.sample_latent(batch_size).to(device))
            img2 = generator2(generator2.sample_latent(batch_size).to(device))
            imgs = torch.cat([img1,img2],dim=0)

            #print(imgs.max(), imgs.min())

            feats = projector(imgs)
        #import pdb; pdb.set_trace()
        logits = dre_model(feats)
        labels = torch.cat([torch.zeros(batch_size), torch.ones(batch_size)],dim=0)
        #labels = F.one_hot(labels.long(),2).float().cuda()

        loss = dre_loss(logits,batch_size)
        if i % 10 == 0:
            acc1 = (logits[:batch_size,0] > logits[:batch_size,1]).sum().item()
            acc2 = ((logits[batch_size:,0] < logits[batch_size:,1]).sum().item())
            acc = (acc2 + acc1)/(batch_size*2)
            print(f"acc: {acc:.4f}, loss {loss}")

            logit_diff = logits[:,0] - logits[:,1]
            _ , new_inds = torch.sort(logit_diff[:batch_size])
            out_imgs = make_grid(img1.clip(0,1)[new_inds]).cpu().detach()
            save_image(out_imgs,f"{i:05d}_1_{acc:.3f}.png")

            logit_diff = logits[:,1] - logits[:,0]
            _ , new_inds = torch.sort(logit_diff[batch_size:])
            out_imgs = make_grid(img2.clip(0,1)[new_inds]).cpu().detach()
            save_image(out_imgs,f"{i:05d}_2_{acc:.3f}.png")

        loss.backward()
        optimizer.step()


    


    torch.save(dre_model.state_dict(),"dre.pt")

    print("done")


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
    #generate a bunch of ws (50)
    #pass them through 1 direction
    #get the diff vector
    #measure cosine sim
    #import pdb; pdb.set_trace()
    '''from sklearn.metrics.pairwise import cosine_similarity
    import matplotlib.pyplot as plt
    ws = generator.sample_latent(16).detach()
    for i in cfg.n_dirs:
        ws_i = model.forward_single(ws, i)
        cosinesi = cosine_similarity((ws_i - ws).cpu().detach().numpy())
        plt.matshow(cosinesi)
        plt.colorbar()
        plt.savefig( os.path.join(hydra.utils.get_original_cwd(), "cosines", f"{i}.png"))
        plt.clf()
        print("direction ", i, "avg similarity ", cosinesi.mean())

    exit()'''
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



from torch.cuda.amp import custom_bwd, custom_fwd

class DifferentiableClamp(torch.autograd.Function):
    """
    In the forward pass this operation behaves like torch.clamp.
    But in the backward pass its gradient is 1 everywhere, as if instead of clamp one had used the identity function.
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, input, max, min):
        if min is None: return input.clamp(max=max)
        if max is None: return input.clamp(min=min)
        return input.clamp(max=max, min = min)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None

def dclamp(input, max=None, min=None):
    """
    Like torch.clamp, but with a constant 1-gradient.
    :param input: The input that is to be clamped.
    :param min: The minimum value of the output.
    :param max: The maximum value of the output.
    """
    return DifferentiableClamp.apply(input, max, min)






@hydra.main(config_path="../conf", config_name="train2")
def hydra_stuff(cfg: DictConfig):
    mymain(cfg)


if __name__ == "__main__":
    hydra_stuff()
