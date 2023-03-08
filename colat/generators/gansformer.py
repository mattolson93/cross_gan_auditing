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

import dnnlib
#import torch

module_path = Path(__file__).parent / "gansformer"
sys.path.insert(1, str(module_path.resolve()))

import loader

class GANsFormerGenerator(Generator):
    def __init__(
        self,
        device: str,
        class_name: str = "ffhq",
        truncation= None,
        use_w: bool = False,
        feature_layer: str = "full",
        num_ws=12,
    ):
        super(GANsFormerGenerator, self).__init__(feature_layer=feature_layer)
        self.device = device
        self.truncation = truncation
        self.latent_avg = None
        self.w_primary = False  # use W as primary latent space?
        self.outclass = class_name


        self.resolution = 256
        self.name = f"GANsFormer-{self.outclass}"
        self.load_model(class_name)

    def n_latent(self):
        return 544

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
        checkpoint = os.path.join(checkpoint_root, f"gansformer/{class_name}.pt")
        self.model = loader.load_network(checkpoint, eval = True)["Gs"].cuda()


   

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

        return self.get_rand_z(n_samples,seed)
        

    def get_max_latents(self):
        return 544

    def set_output_class(self, new_class):
        if self.outclass != new_class:
            raise RuntimeError(
                "GANsFormer: cannot change output class without reloading"
            )

    def save_intermediate(self, layer=""):
        raise RuntimeError(
                "GANsFormer: save_intermediate not implemented"
            )
       

    def get_intermediate(self):
        raise RuntimeError(
                "GANsFormer: get_intermediate not implemented"
            )
        


    def forward(self, x):
        out = self.model(x.reshape(-1,*self.model.input_shape[1:]))[0]
        return 0.5 * (out + 1)

    def custom_partial_forward(self, x, layer_name):
        return self.model(x.reshape(-1,*self.model.input_shape[1:]))[0]


    def partial_forward(self, x, layer_name):
        return self.model(x.reshape(-1,*self.model.input_shape[1:]))[0]

    
