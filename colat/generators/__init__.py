from colat.generators.abstract import Generator
from colat.generators.biggan import BigGANGenerator
from colat.generators.stylegan import StyleGANGenerator
from colat.generators.stylegan2 import StyleGAN2Generator
from colat.generators.proggan import ProgGANGenerator
from colat.generators.gansformer import GANsFormerGenerator
#from colat.generators.sngan_128 import SNGANGenerator
from colat.generators.stylegan3 import StyleGAN3Generator
from colat.generators.styleswin import StyleSwinGenerator


__init__ = [StyleGAN3Generator,StyleSwinGenerator, BigGANGenerator, StyleGANGenerator, StyleGAN2Generator,ProgGANGenerator, GANsFormerGenerator]
