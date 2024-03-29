import torch

from colat.projectors import Projector


class IdentityProjector(Projector, torch.nn.Module):
    def __init__(self, normalize: bool = True,layers: int = 0, load_path=0) -> None:
        net = torch.nn.Sequential(torch.nn.Identity())
        super().__init__(net, normalize)
