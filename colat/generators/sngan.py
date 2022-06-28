from collections import namedtuple
import torch
from torch import nn
import numpy as np
import json
from colat.generators.abstract import Generator


class SNGenerator(Generator):
    def __init__(
        self,
        device: str,
        class_name: str = "mnist",
        feature_layer: str = "temp",
    ):
        self.device = device
        self.feature_layer = feature_layer
        self.class_name = class_name

        configs = {
            # Converted NVIDIA official
            "mnist": 32,
            "cdigit-mnist": 32,
            "cbg-mnist": 32,
        }

        self.resolution = configs[class_name]
        self.config_type = f"sn_resnet{self.resolution}"
        self.load_model()


    def load_model(self):
        checkpoint_root = os.environ.get(
            "GANCONTROL_CHECKPOINT_DIR", Path(__file__).parent / "checkpoints"
        )
        checkpoint = (
            Path(checkpoint_root)
            / f"stylegan2/stylegan2_{self.outclass}_{self.resolution}.pt"
        )

        ckpt = torch.load(checkpoint)

        self.model = make_resnet_generator(SN_RES_GEN_CONFIGS[self.config_type])
        self.model.load_state_dict(ckpt.state_dict(), strict=True)


    def sample_latent(self, n_samples=1, seed=None, truncation=None):
        if seed is None:
            seed = np.random.randint(
                np.iinfo(np.int32).max
            )  # use (reproducible) global rand state

        rng = np.random.RandomState(seed)
        z = (
            torch.from_numpy(
                rng.standard_normal(128 * n_samples).reshape(n_samples, 128)
            )
            .float()
            .to(self.device)
        )  # [N, 512]

        return z

    def get_max_latents(self):
        return 128

    def forward(self,x):
        return self.model(x).view(x.shape[0], 3, self.resolution, self.resolution)

    def partial_forward(self,x, layer_name):
        for i, layer in enumerate(self.model):
            x = layer(x)
            if str(i) == layer_name: return x

        return x.view(x.shape[0], 3, self.resolution, self.resolution)



ResNetGenConfig = namedtuple('ResNetGenConfig', ['channels', 'seed_dim'])
SN_RES_GEN_CONFIGS = {
    'sn_resnet32': ResNetGenConfig([256, 256, 256, 256], 4),
    'sn_resnet64': ResNetGenConfig([16 * 64, 8 * 64, 4 * 64, 2 * 64, 64], 4),
}


MODELS = {
    'sn_resnet32': 32,
    'sn_resnet64': 64,
}


class Args:
    def __init__(self, **kwargs):
        self.nonfixed_noise = False
        self.noises_count = 1
        self.equal_split = False
        self.generator_batch_norm = False
        self.gen_sn = False
        self.distribution_params = "{}"

        self.__dict__.update(kwargs)


def load_model_from_state_dict(root_dir):
    args = Args(**json.load(open(os.path.join(root_dir, 'args.json'))))
    generator_model_path = os.path.join(root_dir, 'generator.pt')

    try:
        image_channels = args.image_channels
    except Exception:
        image_channels = 3

    gen_config = SN_RES_GEN_CONFIGS[args.model]
    generator=  make_resnet_generator(gen_config, channels=image_channels,
                                      latent_dim=args.latent_dim,
                                      img_size=MODELS[args.model])

    generator.load_state_dict(
        torch.load(generator_model_path, map_location=torch.device('cpu')), strict=False)
    return generator


class Reshape(nn.Module):
    def __init__(self, target_shape):
        super(Reshape, self).__init__()
        self.target_shape = target_shape

    def forward(self, input):
        return input.view(self.target_shape)


class ResBlockGenerator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlockGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)

        nn.init.xavier_uniform_(self.conv1.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight.data, np.sqrt(2))

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            self.conv2
            )

        if in_channels == out_channels:
            self.bypass = nn.Upsample(scale_factor=2)
        else:
            self.bypass = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
            )
            nn.init.xavier_uniform_(self.bypass[1].weight.data, 1.0)

    def forward(self, x):
        return self.model(x) + self.bypass(x)




class GenWrapper(nn.Module):
    def __init__(self, model, out_img_shape):
        super(GenWrapper, self).__init__()

        self.model = model
        self.out_img_shape = out_img_shape


    def forward(self, z):
        for layer in self.model:
            z = layer(z)

        img = z
        img = img.view(img.shape[0], *self.out_img_shape)
        return img


def make_resnet_generator(resnet_gen_config, img_size=128, channels=3,
                          latent_dim=128):
    def make_dense():
        dense = nn.Linear(
            distribution.dim, resnet_gen_config.seed_dim**2 * resnet_gen_config.channels[0])
        nn.init.xavier_uniform_(dense.weight.data, 1.)
        return dense

    def make_final():
        final = nn.Conv2d(resnet_gen_config.channels[-1], channels, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(final.weight.data, 1.)
        return final

    model_channels = resnet_gen_config.channels

    input_layers = [
        make_dense(),
        Reshape([-1, model_channels[0], 4, 4])
    ]
    res_blocks = [
        ResBlockGenerator(model_channels[i], model_channels[i + 1])
        for i in range(len(model_channels) - 1)
    ]
    out_layers = [
        nn.BatchNorm2d(model_channels[-1]),
        nn.ReLU(inplace=True),
        make_final(),
        nn.Tanh()
    ]

    model = nn.Sequential(*(input_layers + res_blocks + out_layers))
    return model 
    #return GenWrapper(model, [channels, img_size, img_size], distribution)





