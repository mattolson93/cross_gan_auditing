"""
Implementation of SNGAN for image size 128.
"""
import torch
import torch.nn as nn

from torch_mimicry.modules.layers import SNLinear
from torch_mimicry.modules.resblocks import DBlockOptimized, DBlock, GBlock
from torch_mimicry.nets.sngan import sngan_base

from colat.generators.abstract import Generator
import os
from pathlib import Path
import numpy as np

class SNGANGenerator(Generator):
    def __init__(
        self,
        device: str,
        class_name: str = "sngan",
        truncation: float = 1.0,
        use_w: bool = False,
        feature_layer: str = "full",
        num_ws=12,
    ):
        super(SNGANGenerator, self).__init__(feature_layer=feature_layer)
        self.device = device
        self.truncation = truncation
        self.latent_avg = None
        self.w_primary = False  # use W as primary latent space?
        self.outclass = class_name


        self.resolution = 128
        self.name = f"SNGAN-{self.outclass}"
        self.load_model()

    def n_latent(self):
        return 128

    def latent_space_name(self):
        return "W" if self.w_primary else "Z"
    
    def use_w(self):
        self.w_primary = True

    def use_z(self):
        self.w_primary = False


    def load_model(self):
        checkpoint_root = os.environ.get(
            "GANCONTROL_CHECKPOINT_DIR", Path(__file__).parent / "checkpoints"
        )
        checkpoint = (
            Path(checkpoint_root)
            / f"sngan/snGAN_celeba.pth"
        )

        self.model = SNGANGenerator128().cuda()
        #import pdb; pdb.set_trace()
        self.model.load_state_dict(torch.load(checkpoint, map_location='cuda:0')['model_state_dict'])

   

    @torch.no_grad()
    def truncate_w(self, ws, truncation_val):
        if self.w_primary == False: exit("BAD CALL TO TRUNCATE W")

        w_mins = self.latent_avg - (truncation_val*self.latent_std)
        w_maxs = self.latent_avg + (truncation_val*self.latent_std)

        return torch.min( torch.max(ws, w_mins), w_maxs)


    def get_rand_z(self, n_samples, seed):
        rng = np.random.RandomState(seed)
        z = (
            torch.from_numpy(
                rng.standard_normal(128 * n_samples).reshape(n_samples, 128)
            )
            .float()
            .cuda()
        )  # [N, 512]
        return z

    

    def sample_latent(self, n_samples=1, seed=None, truncation=None):
        if seed is None:
            seed = np.random.randint(
                np.iinfo(np.int32).max
            )  # use (reproducible) global rand state

        return self.get_rand_z(n_samples,seed)
        

    def get_max_latents(self):
        return 128

    def set_output_class(self, new_class):
        if self.outclass != new_class:
            raise RuntimeError(
                "ProgGAN: cannot change output class without reloading"
            )

    def save_intermediate(self, layer=""):
        raise RuntimeError(
                "ProgGAN: save_intermediate not implemented"
            )
       

    def get_intermediate(self):
        raise RuntimeError(
                "ProgGAN: get_intermediate not implemented"
            )
        



    def forward(self, x):
        out = self.model(x)
        return 0.5 * (out + 1)

    def custom_partial_forward(self, x, layer_name):
        return self.model(x)



    def partial_forward(self, x, layer_name):
        return self.model(x)

    



class SNGANGenerator128(sngan_base.SNGANBaseGenerator):
    r"""
    ResNet backbone generator for SNGAN.

    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
        loss_type (str): Name of loss to use for GAN loss.
    """
    def __init__(self, nz=128, ngf=1024, bottom_width=4, **kwargs):
        super().__init__(nz=nz, ngf=ngf, bottom_width=bottom_width, **kwargs)

        # Build the layers
        self.l1 = nn.Linear(self.nz, (self.bottom_width**2) * self.ngf)
        self.block2 = GBlock(self.ngf, self.ngf, upsample=True)
        self.block3 = GBlock(self.ngf, self.ngf >> 1, upsample=True)
        self.block4 = GBlock(self.ngf >> 1, self.ngf >> 2, upsample=True)
        self.block5 = GBlock(self.ngf >> 2, self.ngf >> 3, upsample=True)
        self.block6 = GBlock(self.ngf >> 3, self.ngf >> 4, upsample=True)
        self.b7 = nn.BatchNorm2d(self.ngf >> 4)
        self.c7 = nn.Conv2d(self.ngf >> 4, 3, 3, 1, padding=1)
        self.activation = nn.ReLU(True)

        # Initialise the weights
        nn.init.xavier_uniform_(self.l1.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c7.weight.data, 1.0)

    def forward(self, x):
        r"""
        Feedforwards a batch of noise vectors into a batch of fake images.

        Args:
            x (Tensor): A batch of noise vectors of shape (N, nz).

        Returns:
            Tensor: A batch of fake images of shape (N, C, H, W).
        """
        h = self.l1(x)
        h = h.view(x.shape[0], -1, self.bottom_width, self.bottom_width)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.block6(h)
        h = self.b7(h)
        h = self.activation(h)
        h = torch.tanh(self.c7(h))

        return h

