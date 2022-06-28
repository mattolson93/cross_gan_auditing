from colat.projectors import Projector
from colat.utils.net_utils import create_resnet

import torch
import torch.nn as nn
import torch.nn.functional as F
class CNNProjector(Projector):
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
        

    def forward(self, input: torch.Tensor, which_cnn=1) -> torch.Tensor:
        out = self.net(input,which_cnn)
        if self.normalize:
            norm = torch.norm(out, dim=1)
            return out / torch.reshape(norm, (-1, 1))
        return out


class Net(nn.Module):
    def __init__(self, size, two_cnns):
        super().__init__()
        self.conv1 = nn.Conv2d(512, 512, 5, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(512, size, 3)
        self.poolada = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(size, size)
        self.fc2 = nn.Linear(size, size)
        self.drop = nn.Dropout(p=0.2)

        if two_cnns:
            self.conv12 = nn.Conv2d(512, 512, 5, stride=1)
            self.pool2 = nn.MaxPool2d(2, 2)
            self.conv22 = nn.Conv2d(512, size, 3)
            self.poolada2 = nn.AdaptiveMaxPool2d(1)


    def forward(self, x, which_cnn=1):
        if which_cnn <= 1:
            x = self.pool(F.relu(self.conv1(x)))
            x = self.poolada(F.relu(self.conv2(x)))
        else:
            x = self.pool2(F.relu(self.conv12(x)))
            x = self.poolada2(F.relu(self.conv22(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        #x = self.drop(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x