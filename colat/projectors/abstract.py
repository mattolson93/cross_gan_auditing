from abc import ABC as AbstractBaseClass

import torch
import torchvision.transforms as transforms


class Projector(AbstractBaseClass, torch.nn.Module):
    """Abstract projector

    Args:
        normalize: whether to normalize after feed-forward
    """

    def __init__(self, net: torch.nn.Module, img_preprocess="resize", min_resolution=128, normalize: bool = True) -> None:
        super().__init__()
        self.normalize = normalize
        self.net = net
        self.add_module("Projector", net)
        test_res = 128
        self._resize = transforms.Resize(min(test_res,min_resolution))
        self._crop = transforms.CenterCrop(min(test_res,min_resolution))
        #self._pad = 
        self.img_preprocess = img_preprocess
        #self.net.do_preprocess = self.do_preprocess
        #self.img_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                             std=[0.229, 0.224, 0.225])
        #self.net_



    def do_preprocess(self, x):
        x = dclamp(x, min=0, max=1)
        x = self.transform_preprocess(x)

        return x

    def transform_preprocess(self,x):
        if self.img_preprocess == "resize":
            x =  self._resize(x)
        elif self.img_preprocess == "crop":
            x =  self._crop(x)
        else:
            exit("invalid preprocess type of ", self.img_preprocess)
        return x

    def _clamp(self,x):
        return dclamp(x, min=0, max=1)

    #def img_normalize(self, x):
    #    return self.imgnet_normalize(x)
        
    def get_outsize(self):
        return self.hidden_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = torch.reshape(input, (input.shape[0], -1))
        out = self.net(input)
        if self.normalize:
            norm = torch.norm(out, dim=1)
            return out / torch.reshape(norm, (-1, 1))
        return out


from torch.cuda.amp import custom_bwd, custom_fwd

class DifferentiableClamp(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input, max, min):
        if min is None: return input.clamp(max=max)
        if max is None: return input.clamp(min=min)
        return input.clamp(max=max, min = min)
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None
def dclamp(input, max=None, min=None):
    return DifferentiableClamp.apply(input, max, min)