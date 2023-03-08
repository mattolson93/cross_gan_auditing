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

#import dnnlib
#import torch

module_path = Path(__file__).parent / "styleswin"
sys.path.insert(1, str(module_path.resolve()))
from generator_styleswin import Generator as StyleSwinModel

import torchvision


class StyleSwinGenerator(Generator):
    def __init__(
        self,
        device: str,
        class_name: str = "ffhq",
        truncation= None,
        use_w: bool = False,
        feature_layer: str = "full",
        num_ws=12,
    ):
        super(StyleSwinGenerator, self).__init__(feature_layer=feature_layer)
        self.device = device
        self.truncation = truncation
        self.latent_avg = None
        self.w_primary = False  # use W as primary latent space?
        self.outclass = class_name

        self.inv_normalize = torchvision.transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )


        self.resolution = 256
        self.name = f"StyleSwin-{self.outclass}"
        self.load_model(class_name)

    def n_latent(self):
        return 512

    def latent_space_name(self):
        return "W" if self.w_primary else "Z"
    
    def use_w(self):
        self.w_primary = True

    def use_z(self):
        self.w_primary = False


    def load_model(self, class_name):
        checkpoint_root = os.environ.get(
            "GANCONTROL_CHECKPOINT_DIR", Path(__file__).parent / "checkpoints"
        )
        checkpoint = os.path.join(checkpoint_root, f"styleswin/{class_name}.pt")
        #import pdb; pdb.set_trace()
        ckpt = torch.load(checkpoint)
        if "G_ema" in ckpt.keys():
            ckpt = ckpt['G_ema']
        elif "g_ema" in ckpt.keys():
            ckpt = ckpt['g_ema']
        else:
            ckpt = ckpt['G']


        self.model = StyleSwinModel(
            size=256, style_dim=512, n_mlp=8, channel_multiplier=2, lr_mlp=0.01,
            enable_full_resolution=8, use_checkpoint=True
        ).to(self.device)
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
        if self.w_primary == False: 
            return torch.clamp(ws, min=-truncation_val, max = truncation_val)

        w_mins = self.latent_avg - (truncation_val*self.latent_std)
        w_maxs = self.latent_avg + (truncation_val*self.latent_std)

        return torch.min( torch.max(ws, w_mins), w_maxs)


    def get_rand_z(self, n_samples, seed):
        rng = np.random.RandomState(seed)
        z = (
            torch.from_numpy(
                rng.standard_normal(self.n_latent() * n_samples).reshape(n_samples, self.n_latent())
            )
            .float()
            .to(self.device)
        )  # [N, 512]
        return z

    

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
            z = self.model.style(z)

        return z
        
    def convert_z2w(self, z):
        return self.model.style(z)

    def get_max_latents(self):
        return 512

    def set_output_class(self, new_class):
        if self.outclass != new_class:
            raise RuntimeError(
                "StyleSwinGenerator: cannot change output class without reloading"
            )

    def save_intermediate(self, layer=""):
        raise RuntimeError(
                "StyleSwinGenerator: save_intermediate not implemented"
            )
       

    def get_intermediate(self):
        raise RuntimeError(
                "StyleSwinGenerator: get_intermediate not implemented"
            )
        


    def forward(self, x):
        out = self.model(x, is_z=not self.w_primary)
        return self.inv_normalize(out)

    def custom_partial_forward(self, x, layer_name):
        return self.forward(x)


    def partial_forward(self, x, layer_name):
        return self.forward(x)


    
