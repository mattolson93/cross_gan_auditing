from colat.projectors import Projector

import torch
import torch.nn as nn
import torch.nn.functional as F
class Conv1x1Projector(Projector):
    def __init__(
        self,
        size: int,
        layers: int = 0,
        normalize: bool = True,
    ):
        two_cnns = True
        net = Net(size, two_cnns)
        super().__init__(net, normalize)
        
        # nonlinear mlp
        self.multiconv = True
        

    def forward(self, input: torch.Tensor, which_cnn=1) -> torch.Tensor:
        out = self.net(input,which_cnn)
        if self.normalize:
            norm = torch.norm(out, dim=1)
            return out / torch.reshape(norm, (-1, 1))
        return out


class Net(nn.Module):
    def __init__(self, size, two_cnns):
        super().__init__()
        self.conv1 = nn.Conv2d(512, size, 1, stride=1)

        self.conv2 = nn.Conv2d(512, size, 1, stride=1)
           

    def forward(self, x, which_cnn=1):
        if which_cnn <= 1:
            x = self.conv1(x)
        else:
            x = self.conv2(x)
            #x = x
        #x = F.max_pool2d(x,2)
        x = torch.flatten(x, 1) 
        return x