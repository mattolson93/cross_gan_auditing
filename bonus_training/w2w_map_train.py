import os
import lpips

import sys
# insert at 1, 0 is the script path (or '' in REPL)
#sys.path.insert(1, '../colat')
sys.path.insert(1, '/g/g15/olson60/research/latentclr/')


import hydra
import torch
import torch.nn as nn
from torch.nn import functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from colat2.evaluator import Evaluator
from colat2.projectors import Projector
from colat2.trainer import Trainer
from colat2.visualizer import Visualizer

import numpy as np
import copy

from colat.utils.net_utils import create_dre_model, create_mlp
from torchvision.utils import save_image

def clip_wrapper(losses,clip):
    if clip > 0: losses = torch.max(losses, clip) 
    return losses.mean()

def mymain(cfg: DictConfig) -> None:

    device = get_device(cfg)

    
    # Model
    # Use Hydra's instantiation to initialize directly from the config file
    generator: torch.nn.Module = instantiate(cfg.generator).to(device)
    generator2: torch.nn.Module = instantiate(cfg.generator2).to(device)

    mat_map = torch.nn.Linear(512,512, bias=False).to(device)
    #import pdb; pdb.set_trace()
    torch.nn.init.eye_(mat_map.weight)

    print(mat_map.weight)

    optimizer: torch.optim.Optimizer = instantiate(
        cfg.hparams.optimizer,
        list(mat_map.parameters()),
    )

    batch_size = cfg.hparams.batch_size

    generator.eval()
    generator2.eval()
    percept = lpips.LPIPS(net='vgg', spatial=True).to(generator.device)

    l2_lam = cfg.l2_lam
    mse_lam = cfg.mse_lam
    p_lamb = cfg.p_lam
    clip_val = cfg.clip
    clip_val = torch.tensor(clip_val, dtype=torch.float).cuda()


    pbar = tqdm(range(cfg.hparams.iterations))
    pixel_loss1_avg = []
    pixel_loss2_avg = []
    perc_loss1_avg = []
    perc_loss2_avg = []

    for i in pbar:
        optimizer.zero_grad()
        with torch.no_grad():
            ws1 = generator.sample_latent(batch_size).to(device)
        mapped_ws2 = mat_map(ws1)
        img1 = generator(ws1)
        img2 = generator2(mapped_ws2)

        


        #latent_loss1=  clip_wrapper(F.mse_loss(ws1,  ws1_pred, reduction='none'), clip_val)
        #latent_loss2=  clip_wrapper(F.mse_loss(ws2,  ws2_pred, reduction='none'), clip_val )
        pixel_loss =  F.mse_loss(img1, img2)
        #perc_loss =   percept(img1, img2).mean()
        
        if i > (cfg.hparams.iterations - 50):
            with torch.no_grad():
                pixel_loss_avg.append(F.mse_loss(img1, img2).mean().item())
                #perc_loss_avg.append(percept(img1, img2).mean().item())


        loss = mse_lam*pixel_loss #+ p_lamb* perc_loss
        
        loss.backward()

        if i%10 == 0:
            #import pdb; pdb.set_trace()
            #print(f'pixel_loss: {pixel_loss.item():.4f} perc_loss: {perc_loss.item():.4f}')
            print(f'pixel_loss: {pixel_loss.item():.4f}')

        #pbar.set_description( f'pixel_loss: {pixel_loss.item():.4f} perc_loss: {perc_loss.item():.4f}')
        pbar.set_description( f'pixel_loss: {pixel_loss.item():.4f}')
        optimizer.step()
        

    pixel_loss1_avg = np.array(pixel_loss1_avg).mean()
    pixel_loss2_avg = np.array(pixel_loss2_avg).mean()
    perc_loss1_avg = np.array(perc_loss1_avg).mean()
    perc_loss2_avg = np.array(perc_loss2_avg).mean()

    pixel_loss = (pixel_loss1_avg  + pixel_loss2_avg)/2
    perc_loss  = (perc_loss1_avg   + perc_loss2_avg)/2

    sumed = latent_loss + pixel_loss + perc_loss





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

from torchvision import models
import torchvision.transforms as transforms
class AttClsModel(torch.nn.Module):
    def __init__(self, model_type, layers, is_pretrained=True, dropout=0.5):
        super(AttClsModel, self).__init__()
        
        self.backbone = models.resnet18(pretrained=is_pretrained)
        self.backbone.fc = torch.nn.Identity() 
        hidden_size = 512
       
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        self.resize = transforms.Resize(128)
        self.dropout = torch.nn.Dropout(dropout)

        self.fc_dataset1 = create_mlp(2,hidden_size, hidden_size*2,hidden_size, batchnorm=False)
        self.fc_dataset2 = create_mlp(2,hidden_size, hidden_size*2,hidden_size, batchnorm=False)
        self.leaky = nn.LeakyReLU(0.2)



    def preprocess_img(self, img):
        typical_img = (img).clamp(0, 1)
        return self.resize(self.normalize(typical_img))


    def forward(self, input, labels=None):
        x = self.backbone(self.preprocess_img(input))

        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.leaky(x)



        return self.fc_dataset1(x), self.fc_dataset2(x)


@hydra.main(config_path="../conf", config_name="train_encoder")
def hydra_stuff(cfg: DictConfig):
    mymain(cfg)


if __name__ == "__main__":
    hydra_stuff()
