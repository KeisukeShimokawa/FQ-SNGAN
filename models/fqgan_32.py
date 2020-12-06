"""
implementation ideas from https://github.com/kwotsin/mimicry/blob/master/torch_mimicry/nets/sngan/sngan_32.py
"""
import torch
import torch.nn as nn

from modules.layers import SNLinear
from modules.resblocks import GBlock, DBlockOptimized, DBlock
from modules.utils import init_weight
from modules.vq import  FeatureQuantizer, FeatureQuantizerEMA


class FQGANGenerator32(nn.Module):
    r"""
    ResNet backbone generator for SNGAN-based FQGAN.

    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
        init_mode (str): initialization mode in ["ortho", "N02", "xavier", "glorot"].
    """
    def __init__(self, nz=128, ngf=256, bottom_width=4, init_mode="xavier"):
        super().__init__()
        self.nz = nz
        self.ngf = ngf
        self.bottom_width = bottom_width

        self.l1 = nn.Linear(self.nz, (self.bottom_width * self.bottom_width) * self.ngf)
        self.block2 = GBlock(self.ngf, self.ngf, upsample=True)
        self.block3 = GBlock(self.ngf, self.ngf, upsample=True)
        self.block4 = GBlock(self.ngf, self.ngf, upsample=True)
        self.b5 = nn.BatchNorm2d(self.ngf)
        self.c5 = nn.Conv2d(self.ngf, 3, kernel_size=3, stride=1, padding=1)
        self.activation = nn.ReLU(inplace=True)

        # weight init
        init_weight(self.modules, init_mode)

    def forward(self, x):
        r"""
        Feedforwards a batch of noise vectors into a batch of fake images.

        Args:
            x (Tensor): A batch of noise vectors of shape (N, nz).

        Returns:
            Tensor: A batch of fake images of shape (N, C, H, W).
        """
        h = self.l1(x)
        h = h.view(-1, self.ngf, self.bottom_width, self.bottom_width)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.b5(h)
        h = self.activation(h)
        h = self.c5(h)
        output = torch.tanh(h)

        return output


class FQGANDiscriminator32(nn.Module):
    r"""
    ResNet backbone discriminator for SNGAN-based FQGAN.

    Attributes:
        ndf (int): Variable controlling discriminator feature map sizes.
        init_mode (str): initialization mode in ["ortho", "N02", "xavier", "glorot"].
        fq_type (str): Embedding Learning Strategy within ['Normal', 'EMA']
        dict_size (int): Number of discrete vector on codebook by 2**p. default to 10.
        quant_layers (int, list[int]): Position of applying feature quantization module.
        fq_strength (float): Weight of feature quantization module
    """
    def __init__(self,
                 ndf=128,
                 init_mode="xavier",
                 fq_type=None,
                 dict_size=10,
                 quant_layers=None,
                 fq_strength=10.0):
        super().__init__()
        self.ndf = ndf
        self.fq_type = fq_type
        self.dict_size = dict_size
        self.quant_layers = quant_layers
        self.fq_strength = fq_strength

        self.block1 = DBlockOptimized(3, self.ndf)
        self.block2 = DBlock(self.ndf, self.ndf, downsample=True)
        self.block3 = DBlock(self.ndf, self.ndf, downsample=False)
        self.block4 = DBlock(self.ndf, self.ndf, downsample=False)
        self.l5 = SNLinear(self.ndf, 1)
        self.activation = nn.ReLU(inplace=True)

        if isinstance(self.quant_layers, int):
            self.quant_layers = [self.quant_layers]

        if self.fq_type is not None:
            assert self.fq_type in ['Normal', 'EMA'], "set fq_type within ['Normal', 'EMA']"

        if self.fq_type:
            assert self.quant_layers is not None, "should set quant_layers like ['3']"
            assert (min(self.quant_layers) > 1) and (max(self.quant_layers) < 5), "should be range [2, 4]"
            for layer in self.quant_layers:
                out_channels = getattr(self, f"block{layer}").out_channels
                if self.fq_type == "Normal":
                    setattr(self, f"fq{layer}", FeatureQuantizer(out_channels, 2 ** self.dict_size))
                elif self.fq_type == "EMA":
                    setattr(self, f"fq{layer}", FeatureQuantizerEMA(out_channels, 2 ** self.dict_size))

        init_weight(self.modules, init_mode)

    def forward(self, x):
        r"""
        Feedforwards a batch of real/fake images and produces a batch of GAN logits.

        Args:
            x (Tensor): A batch of images of shape (N, C, H, W).

        Returns:
            Tensor: A batch of GAN logits of shape (N, 1).
            Tensor: All loss of each FQ module.
            List[Tensor]:  All embedding index of each FQ module.
        """
        h = x
        h = self.block1(h)

        # compute quant layer
        all_quant_loss = 0
        all_embed_idx = None
        for layer in range(2, 5):
            h = getattr(self, f"block{layer}")(h)
            # apply Feature Quantization
            if (self.fq_type) and (layer in self.quant_layers):
                h, loss, embed_idx = getattr(self, f"fq{layer}")(h)
                all_quant_loss += loss
                all_embed_idx.append(embed_idx)

        h = self.activation(h)
        # Global sum pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.l5(h)

        return output, all_quant_loss, all_embed_idx
