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


class ProgGANGenerator(Generator):
    def __init__(
        self,
        device: str,
        class_name: str = "ffhq",
        truncation: float = 1.0,
        use_w: bool = False,
        feature_layer: str = "full",
        num_ws=12,
    ):
        super(ProgGANGenerator, self).__init__(feature_layer=feature_layer)
        self.device = device
        self.truncation = truncation
        self.latent_avg = None
        self.w_primary = False  # use W as primary latent space?
        self.outclass = class_name


        self.resolution = 256
        self.name = f"ProgGAN-{self.outclass}"
        self.load_model()

    def n_latent(self):
        return 512

    def latent_space_name(self):
        return "W" if self.w_primary else "Z"
    
    def use_w(self):
        self.w_primary = True

    def use_z(self):
        self.w_primary = False


    def load_model(self):
        self.model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                       'PGAN', model_name='celebAHQ-512',
                       pretrained=True, useGPU=True)


   

    @torch.no_grad()
    def truncate_w(self, ws, truncation_val):
        return torch.clamp(ws, min=-truncation_val, max=truncation_val )



    def get_rand_z(self, n_samples, seed):
        torch.manual_seed(seed)
        z, _ = self.model.buildNoiseData(n_samples)
       
        return z.cuda()

    

    def sample_latent(self, n_samples=1, seed=None, truncation=None):
        if seed is None:
            seed = np.random.randint(
                np.iinfo(np.int32).max
            )  # use (reproducible) global rand state

        return self.get_rand_z(n_samples,seed)
        

    def get_max_latents(self):
        return 512

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
        out = self.model.test(x, getAvG=True, toCPU=False)
        return 0.5 * (out + 1)

    def custom_partial_forward(self, x, layer_name):
        return self.forward(x)



    def partial_forward(self, x, layer_name):
        return self.forward(x)

    
