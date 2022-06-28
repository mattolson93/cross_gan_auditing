from colat.projectors import Projector
from colat.utils.net_utils import create_resnet

import torch
class ResNetProjector(Projector):
    def __init__(
        self,
        name: str,
        layers: int,
        load_path: str = None,
        normalize: bool = True,
        
    ):
        # nonlinear mlp
        self.layers=layers
        
        net = create_resnet(name=name, layers=layers, load_path=load_path)
        super().__init__(net, normalize)
        self.hidden_size = net.hidden_size

    def get_size(self): return self.hidden_size
    

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.net(input)
        if self.normalize:
            norm = torch.norm(out, dim=1)
            return out / torch.reshape(norm, (-1, 1))
        return out
