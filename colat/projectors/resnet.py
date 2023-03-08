from colat.projectors import Projector
from colat.utils.net_utils import create_resnet
import torchvision.transforms as transforms

import torch
class ResNetProjector(Projector):
    def __init__(
        self,
        name: str,
        layers: int,
        load_path: str = None,
        normalize: bool = True,
        img_preprocess="resize", 
        min_resolution=128, 
    ):
        # nonlinear mlp
        self.layers=layers
        
        net, possible_transform = create_resnet(name=name, layers=layers, load_path=load_path)
        super().__init__(net, img_preprocess, min_resolution, normalize)
        self.hidden_size = net.hidden_size

        if possible_transform is not None: 
            self.transform_normalize = possible_transform
        else:
            self.transform_normalize =  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        

    def get_size(self): return self.hidden_size
    

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        #import pdb; pdb.set_trace()
        prepped_img = self.do_preprocess(input)
        prepped_img = self.transform_normalize(prepped_img)

        out = self.net(prepped_img)
        if self.normalize:
            norm = torch.norm(out, dim=1)
            return out / torch.reshape(norm, (-1, 1))
        return out
