from colat.projectors import Projector
import torch
import clip

from transformers import CLIPProcessor, CLIPModel
from torchvision.transforms import Resize, Normalize
import torchvision

class ClipProjector(Projector):
    def __init__(
        self,
        layers: int,
        normalize: bool = True,
        name="None",
        load_path="",
        img_preprocess="resize", 
        min_resolution=128, 
    ):
        self.layers=layers

        self.name = name

        prepro_kwargs = {
            "do_resize" : False,
            "do_convert_rgb" : False,
            "do_center_crop" : False,
            "do_normalize" : False,
        }

        if name == "rsicd":
            exit("rsicd doesnt work, fix later")
            net = CLIPModel.from_pretrained("/usr/WS2/olson60/research/latentclr/clip-rsicd-v2").cuda()
            preprocess = CLIPProcessor.from_pretrained("/usr/WS2/olson60/research/latentclr/clip-rsicd-v2", **prepro_kwargs)
            self.hidden_size =  net.vision_embed_dim
        else:
            #import pdb; pdb.set_trace()
            net, preprocess = clip.load(name.replace("_", "/"), device="cuda")
            #del preprocess.transforms[2] #remove PIL load
            #del preprocess.transforms[2] #remove ToTensor
            #import pdb; pdb.set_trace()
            self.hidden_size = net.visual.output_dim


        super().__init__(net, img_preprocess, max(224,min_resolution), normalize)

        #replace the base class imgtransforms
        self.transform_normalize = preprocess.transforms[-1]
        
        self._resize = Resize(224, torchvision.transforms.InterpolationMode.BICUBIC)
        
        self.net = net
        self.preprocess = preprocess

    def get_size(self): return self.hidden_size

    '''def process_img(self, input):
        input = dclamp((0.5 * (input + 1)), min=0, max=1)

        if self.name == "rsicd":
            img = self.resize(self.normalize(input))
            inputs = self.preprocess(images=img, return_tensors="pt")
            return self.net.get_image_features(**inputs)
        else:
            image = self.preprocess(input)
            return self.net.encode_image(image)
    '''


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        img = self.do_preprocess(input)
        img = self.transform_normalize(img)
        
        out = self.net.encode_image(img)

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