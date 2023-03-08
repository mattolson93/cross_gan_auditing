from colat.projectors import Projector

import torch
from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
#from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torchvision.transforms as transforms
import hydra
import os


class VitProjector(Projector):
    def __init__(
        self,
        layers: int,
        normalize: bool = True,
        name = "",
        load_path = "",
        img_preprocess="resize", 
        min_resolution=128, 
    ):
        self.layers=layers
        pretrained_deit = load_path==""
        net  = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=pretrained_deit)
        self.transform_normalize =  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if load_path !="":
            #import pdb; pdb.set_trace()
            pt_path = os.path.join( hydra.utils.get_original_cwd(), load_path)
            net.load_state_dict(torch.load(pt_path)['model'],strict=False) #False because we don't have a prediction head

        self.hidden_size = 768

        super().__init__(net, img_preprocess, min_resolution, normalize)

    def transform(self):
        pass

    def get_size(self): return self.hidden_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        #import pdb; pdb.set_trace()
        input = self.do_preprocess(input)
        input = self.transform_normalize(input)
        
        out = self.net.forward_features(input)

        if self.normalize:
            norm = torch.norm(out, dim=1)
            return out / torch.reshape(norm, (-1, 1))
        return out

