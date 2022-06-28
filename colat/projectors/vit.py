from colat.projectors import Projector

import torch
from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torchvision.transforms as transforms

class VitProjector(Projector):
    def __init__(
        self,
        layers: int,
        normalize: bool = True,
        
    ):
        self.layers=layers
        
        net  = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])

        self.hidden_size = 768

        super().__init__(net, normalize)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        #import pdb; pdb.set_trace()
        input = self.transform(input)
        out = self.net.forward_features(input)

        if self.normalize:
            norm = torch.norm(out, dim=1)
            return out / torch.reshape(norm, (-1, 1))
        return out

