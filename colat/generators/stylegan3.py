# Copyright 2020 Erik Härkönen. All rights reserved.
# This file is licensed to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License. You may obtain a copy
# of the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

import os
import sys
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from colat.generators.abstract import Generator

module_path = Path(__file__).parent / "stylegan3"
sys.path.insert(1, str(module_path.resolve()))



from networks_stylegan3 import Generator as StyleGAN3Model


class StyleGAN3Generator(Generator):
    def __init__(
        self,
        device: str,
        class_name: str = "ffhq",
        truncation: float = 1.0,
        use_w: bool = False,
        feature_layer: str = "generator.layers.0",
        num_ws=12,
    ):
        super(StyleGAN3Generator, self).__init__(feature_layer=feature_layer)
        self.device = device
        self.truncation = truncation
        self.latent_avg = None
        self.w_primary = use_w  # use W as primary latent space?
        self.outclass = class_name

        # Image widths
        configs = {
            # Converted NVIDIA official
            "ffhqu-r": 256,
            "ffhqu-t": 256,
        }

        self.resolution = configs[self.outclass]
        self.name = f"StyleGAN3-{self.outclass}"
        self.load_model()
        self.set_noise_seed(0)

    def n_latent(self): return 512
    def latent_space_name(self): return "W" if self.w_primary else "Z"

    def use_w(self): self.w_primary = True
    def use_z(self): self.w_primary = False


    def load_model(self):
        checkpoint_root = os.environ.get(
            "GANCONTROL_CHECKPOINT_DIR", Path(__file__).parent / "checkpoints"
        )
        checkpoint = ( Path(checkpoint_root)  / f"stylegan3/{self.outclass}_{self.resolution}.pt" )


        
        #import pdb; pdb.set_trace()
        #with open(fname, 'rb') as f: obj = f.read()
        #ckpt = pickle.loads(obj, encoding='latin1')
        ckpt = torch.load(checkpoint)
        '''if "G_ema" in ckpt.keys():
            ckpt = ckpt['G_ema']
        elif "g_ema" in ckpt.keys():
            ckpt = ckpt['g_ema']
        else:
            ckpt = ckpt['G']'''



        if "-t" in self.outclass:
            G_kwargs = {'z_dim': 512, 'w_dim': 512, 'mapping_kwargs': {'num_layers': 2}, 'channel_base': 16384 , 'channel_max': 512, 'magnitude_ema_beta': 0.9997227795604651,'c_dim': 0, 'img_resolution': 256, 'img_channels': 3}
        elif "-r" in self.outclass:
            #G_kwargs = {'z_dim': 512, 'w_dim': 512, 'mapping_kwargs': {'num_layers': 2}, 'channel_base': 16384 , 'channel_max': 1024, 'magnitude_ema_beta': 0.9997227795604651, 'conv_kernel': 1, 'use_radial_filters': True, 'c_dim': 0, 'img_resolution': 256, 'img_channels': 3}
            G_kwargs = {'z_dim': 512, 'w_dim': 512, 'mapping_kwargs': {'num_layers': 2}, 'channel_base': (16384*2) , 'channel_max': 1024, 'magnitude_ema_beta': 0.9997227795604651, 'conv_kernel': 1, 'use_radial_filters': True, 'c_dim': 0, 'img_resolution': 256, 'img_channels': 3}
        else: 
            ValueError("Bad stylgan3 classname")

        #import pdb; pdb.set_trace()
        self.model = StyleGAN3Model(**G_kwargs)

        
        self.model = self.model.to(self.device)
        self.model.load_state_dict(ckpt)

        self.latent_avg, self.latent_std = self.calc_latent_avg()

    @torch.no_grad()
    def calc_latent_avg(self):
        seed = 0
        w = []
        for i in range(100): w.extend(self.convert_z2w(self.get_rand_z(500, seed)))

        stacked_w = torch.stack(w,dim=0)
        w_mean = stacked_w.mean(0)
        w_std  = stacked_w.std(0)

        return w_mean, w_std

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
                rng.standard_normal(512 * n_samples).reshape(n_samples, 512)
            )
            .float()
            .to(self.device)
        )  # [N, 512]
        return z

    def convert_z2w(self, z):
        return self.model.mapping.forward_w_only(z, None)

    def sample_latent(self, n_samples=1, seed=None, truncation=None):
        if seed is None:
            seed = np.random.randint(
                np.iinfo(np.int32).max
            )  # use (reproducible) global rand state

        rng = np.random.RandomState(seed)
        z = (
            torch.from_numpy(
                rng.standard_normal(512 * n_samples).reshape(n_samples, 512)
            )
            .float()
            .to(self.device)
        )  # [N, 512]

        if self.w_primary:
            z = self.model.mapping.forward_w_only(z, None)

        return z

    def get_max_latents(self):
        return self.model.n_latent

    def set_output_class(self, new_class):
        if self.outclass != new_class:
            raise RuntimeError(
                "StyleGAN3: cannot change output class without reloading"
            )

    def save_intermediate(self, layer=""):
        raise RuntimeError(
                "StyleGAN3: save_intermediate not implemented"
            )
       

    def get_intermediate(self):
        raise RuntimeError(
                "StyleGAN3: get_intermediate not implemented"
            )



    def forward(self, x, **kwargs):
        
        if self.w_primary:
            x = self.model.mapping.forward_w_broadcast(x)
            out = self.model.synthesis(x, **kwargs)
        else:
            out = self.model(x,c=None)
        
        return 0.5 * (out + 1)

    def custom_partial_forward(self, x, layer_name, **kwargs):
        return self.forward(x)


    def partial_forward(self, x, layer_name, **kwargs):
        return self.forward(x)

    def set_noise_seed(self, seed):
        torch.manual_seed(seed)
       
