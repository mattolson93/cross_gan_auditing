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

module_path = Path(__file__).parent / "stylegan2-pytorch"
sys.path.insert(1, str(module_path.resolve()))

module_path = Path(__file__).parent / "stylegan2-ada"
sys.path.insert(1, str(module_path.resolve()))

#import pdb; pdb.set_trace()

from model import Generator as StyleGAN2Model
from networks import Generator as StyleGAN2Model2
from splicenetworks import Generator as StyleGAN2Model2spliced

from colat.utils.model_utils import download_ckpt


class StyleGAN2Generator(Generator):
    def __init__(
        self,
        device: str,
        class_name: str = "ffhq",
        truncation: float = 1.0,
        use_w: bool = False,
        feature_layer: str = "generator.layers.0",
        num_ws=12,
        do_splice=False,
    ):
        super(StyleGAN2Generator, self).__init__(feature_layer=feature_layer)
        self.device = device
        self.truncation = truncation
        self.latent_avg = None
        self.w_primary = use_w  # use W as primary latent space?
        self.outclass = class_name

        # Image widths
        configs = {
            # Converted NVIDIA official
            "car": 512,
            "cat": 256,
            "church": 256,
            "horse": 256,
            "anime": 512,
            # Tuomas
            "bedrooms": 256,
            "kitchen": 256,
            "places": 256,
            #new dl from nvidia
            "ffhq": 1024,
            "ffhqsmall": 256,
            "metfacesu": 1024 ,
            "metfaces": 1024 ,
            "lsundog": 256 ,
            "ffhqu": 1024 ,
            "ffhqusmall": 256 ,
            "celebahq": 256 ,
            "brecahad": 512 ,
            "afhqwild": 512 ,
            "afhqv2": 512 ,
            "afhqdog": 512 ,
            "afhqcat": 512 ,
            #https://github.com/happy-jihye/Cartoon-StyleGAN
            "toon": 256 ,
            "disney": 256 ,
            "metfacesmall": 256 ,
        }
        new_downloaded = ["ffhq","ffhqsmall","ffhqusmall","metfacesu", "metfaces", "lsundog", "ffhqu", "celebahq", "brecahad", "afhqwild", "afhqv2", "afhqdog", "afhqcat"]

        #assert (
        #    self.outclass in configs
        #), f'Invalid StyleGAN2 class {self.outclass}, should be one of [{", ".join(configs.keys())}]'

        self.resolution = 128 if class_name not in configs else configs[self.outclass]
        self.custom_stylegan2 = class_name not in configs or class_name in new_downloaded
        self.name = f"StyleGAN2-{self.outclass}"
        self.has_latent_residual = True
        self.load_model()
        self.set_noise_seed(0)

    def n_latent(self):
        return self.model.n_latent

    def latent_space_name(self):
        return "W" if self.w_primary else "Z"

    def use_w(self):
        self.w_primary = True

    def use_z(self):
        self.w_primary = False

    # URLs created with https://sites.google.com/site/gdocs2direct/
    def download_checkpoint(self, outfile):
        checkpoints = {
            "horse": "https://drive.google.com/uc?export=download&id=18SkqWAkgt0fIwDEf2pqeaenNi4OoCo-0",
            "ffhq": "https://drive.google.com/uc?export=download&id=1FJRwzAkV-XWbxgTwxEmEACvuqF5DsBiV",
            "church": "https://drive.google.com/uc?export=download&id=1HFM694112b_im01JT7wop0faftw9ty5g",
            "car": "https://drive.google.com/uc?export=download&id=1iRoWclWVbDBAy5iXYZrQnKYSbZUqXI6y",
            "cat": "https://drive.google.com/uc?export=download&id=15vJP8GDr0FlRYpE8gD7CdeEz2mXrQMgN",
            "places": "https://drive.google.com/uc?export=download&id=1X8-wIH3aYKjgDZt4KMOtQzN1m4AlCVhm",
            "bedrooms": "https://drive.google.com/uc?export=download&id=1nZTW7mjazs-qPhkmbsOLLA_6qws-eNQu",
            "kitchen": "https://drive.google.com/uc?export=download&id=15dCpnZ1YLAnETAPB0FGmXwdBclbwMEkZ",
        }

        url = checkpoints[self.outclass]
        download_ckpt(url, outfile)

    def load_model(self):
        checkpoint_root = os.environ.get(
            "GANCONTROL_CHECKPOINT_DIR", Path(__file__).parent / "checkpoints"
        )
        checkpoint = (
            Path(checkpoint_root)
            / f"stylegan2/stylegan2_{self.outclass}_{self.resolution}.pt"
        )


        #import pdb; pdb.set_trace()
        if self.custom_stylegan2:

            ckpt = torch.load(checkpoint)
            if "G_ema" in ckpt.keys():
                ckpt = ckpt['G_ema']
            elif "g_ema" in ckpt.keys():
                ckpt = ckpt['g_ema']
            else:
                ckpt = ckpt['G']

            z_dim = ckpt.z_dim
            c_dim = ckpt.c_dim
            w_dim = ckpt.w_dim
            img_resolution = ckpt.img_resolution
            img_channels = ckpt.img_channels

            mapping_kwargs = {"num_layers":ckpt.mapping.num_layers, "w_avg_beta": ckpt.mapping.w_avg_beta}

            synthesis_kwargs = {"channel_base": ckpt.synthesis.__dict__['_init_kwargs']['channel_base']}
            if "conv_clamp" in ckpt.synthesis.__dict__['_init_kwargs'].keys():
                synthesis_kwargs["conv_clamp"] = ckpt.synthesis.__dict__['_init_kwargs']['conv_clamp'] 

            do_splice = False
            if do_splice:
                self.model = StyleGAN2Model2spliced(z_dim, c_dim, w_dim, img_resolution, img_channels, mapping_kwargs=mapping_kwargs, synthesis_kwargs=synthesis_kwargs).to(self.device)
            else:
                self.model = StyleGAN2Model2(z_dim, c_dim, w_dim, img_resolution, img_channels, mapping_kwargs=mapping_kwargs, synthesis_kwargs=synthesis_kwargs).to(self.device)
            self.model.load_state_dict(ckpt.state_dict(), strict=True)
            '''if self.w_primary:
                ws = []
                for i in range(100):
                    ws.append(self.sample_latent(n_samples=100))
                #import pdb; pdb.set_trace()
                ws = torch.cat(ws, dim=0) 
                print(ws.mean(0)[:10])
                print(ws.mean(0)[-10:])
                print(ws.std(0)[:10])
                print(ws.std(0)[-10:])
                exit()
            '''

            #self.model = ckpt['G'].to(self.device)
            self.model.log_size = int(np.log2(img_resolution))
            self.latent_avg = self.model.mapping.w_avg.to(self.device)
            self.model.synthesis.save_block = ''

        else:
            self.model = StyleGAN2Model(self.resolution, 512, 8).to(self.device)
            if not checkpoint.is_file():
                os.makedirs(checkpoint.parent, exist_ok=True)
                print("you'll need to manually download the following file")
                self.download_checkpoint(checkpoint)
                
            ckpt = torch.load(checkpoint)

            if self.outclass =="anime":
                self.latent_avg = ckpt["truncation_latent"].to(self.device)
            elif self.outclass =='toon' or self.outclass =='metfacesmall' or self.outclass =='disney':
                ckpt = ckpt["g_ema"]
            else:
                self.latent_avg = ckpt["latent_avg"].to(self.device)
                ckpt = ckpt["g_ema"]

            self.model.load_state_dict(ckpt, strict=False)
            self.save_block = ''

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
        if self.custom_stylegan2:
            #import pdb; pdb.set_trace()
            z = self.model.mapping.forward_w_only(z, None)
        else:
            z = self.model.style(z)
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
            if self.custom_stylegan2 :
                #import pdb; pdb.set_trace()
                z = self.model.mapping.forward_w_only(z, None)
            else:
                z = self.model.style(z)

        return z

    def get_max_latents(self):
        return self.model.n_latent

    def set_output_class(self, new_class):
        if self.outclass != new_class:
            raise RuntimeError(
                "StyleGAN2: cannot change output class without reloading"
            )

    def save_intermediate(self, layer=""):
        if self.custom_stylegan2:
            self.model.synthesis.save_block = 'block_2' if layer == "" else layer
        else:
            self.save_block = 'convs.2'  if layer == "" else layer

    def get_intermediate(self):
        if self.custom_stylegan2:
            return self.model.synthesis.save_block_output
        else:
            return self.model.save_block_output



    def forward(self, x):
        
        x = x if isinstance(x, list) or self.custom_stylegan2 else [x]
        if self.w_primary and self.custom_stylegan2:
            x = self.model.mapping.forward_w_broadcast(x)
            out = self.model.synthesis(x)
        elif self.custom_stylegan2:
            out = self.model(x,c=None)
        else:
            out, _ = self.model(
                x,
                noise=self.noise,
                truncation=1,
                truncation_latent=self.latent_avg,
                input_is_w=self.w_primary,
            )
        return 0.5 * (out + 1)

    def custom_partial_forward(self, x, layer_name):
        self.model.synthesis.block_out = layer_name
        if self.w_primary:
            return self.model.synthesis(self.model.mapping.forward_w_broadcast(x), block_out=layer_name)


        return self.model(x, c=None, block_out=layer_name)



    def partial_forward(self, x, layer_name):
        if self.custom_stylegan2: 
            return self.custom_partial_forward(x, layer_name)

        styles = x if isinstance(x, list) else [x]
        inject_index = None
        noise = self.noise

        if not self.w_primary:
            styles = [self.model.style(s) for s in styles]

        if len(styles) == 1:
            # One global latent
            inject_index = self.model.n_latent
            latent = self.model.strided_style(
                styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            )  # [N, 18, 512]
        elif len(styles) == 2:
            # Latent mixing with two latents
            if inject_index is None:
                inject_index = random.randint(1, self.model.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = (
                styles[1].unsqueeze(1).repeat(1, self.model.n_latent - inject_index, 1)
            )

            latent = self.model.strided_style(torch.cat([latent, latent2], 1))
        else:
            # One latent per layer
            assert (
                len(styles) == self.model.n_latent
            ), f"Expected {self.model.n_latents} latents, got {len(styles)}"
            styles = torch.stack(styles, dim=1)  # [N, 18, 512]
            latent = self.model.strided_style(styles)

        if "style" in layer_name:
            return latent

        out = self.model.input(latent)
        if "input" == layer_name:
            return out

        out = self.model.conv1(out, latent[:, 0], noise=noise[0])
        if "conv1" in layer_name:
            return out

        skip = self.model.to_rgb1(out, latent[:, 1])
        if "to_rgb1" in layer_name:
            return skip

        i = 1
        noise_i = 1

        for conv1, conv2, to_rgb in zip(
            self.model.convs[::2], self.model.convs[1::2], self.model.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise[noise_i])
            if f"convs.{i-1}" in layer_name:
                return out

            if f"convs.{i-1}" in self.save_block: self.save_block_output = out

            out = conv2(out, latent[:, i + 1], noise=noise[noise_i + 1])
            if f"convs.{i}" in layer_name:
                return out
            
            if f"convs.{i}" in self.save_block: self.save_block_output = out

            skip = to_rgb(out, latent[:, i + 2], skip)
            if f"to_rgbs.{i//2}" in layer_name:
                return skip


            i += 2
            noise_i += 2

        #import pdb; pdb.set_trace()
        if "full" == layer_name:
            return skip

        raise RuntimeError(f"Layer {layer_name} not encountered in partial_forward")
        '''outimg = ((skip + 1)/2).clip(0,1)
        os.chdir("/usr/WS2/olson60/research/latentclr/")
        from torchvision.utils import save_image
        from torchvision.utils import make_grid
        save_image(make_grid(outimg.cpu().detach()),"test.png")'''

    def set_noise_seed(self, seed):
        torch.manual_seed(seed)
        self.noise = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=self.device)]

        for i in range(3, self.model.log_size + 1):
            for _ in range(2):
                self.noise.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=self.device))
