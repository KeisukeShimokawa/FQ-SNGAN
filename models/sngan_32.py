r"""
implementation ideas from https://github.com/kwotsin/mimicry/blob/master/torch_mimicry/nets/sngan/sngan_32.py
"""
import torch
import torch.nn as nn

from modules.layers import SNLinear
from modules.resblocks import GBlock, DBlockOptimized, DBlock
from modules.utils import init_weight


class SNGANGenerator32(nn.Module):
    r"""
    ResNet backbone generator for SNGAN.

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


class SNGANDiscriminator32(nn.Module):
    r"""
    ResNet backbone discriminator for SNGAN.

    Attributes:
        ndf (int): Variable controlling discriminator feature map sizes.
        init_mode (str): initialization mode in ["ortho", "N02", "xavier", "glorot"].
    """
    def __init__(self,
                 ndf=128,
                 init_mode="xavier"):
        super().__init__()
        self.ndf = ndf

        self.block1 = DBlockOptimized(3, self.ndf)
        self.block2 = DBlock(self.ndf, self.ndf, downsample=True)
        self.block3 = DBlock(self.ndf, self.ndf, downsample=False)
        self.block4 = DBlock(self.ndf, self.ndf, downsample=False)
        self.l5 = SNLinear(self.ndf, 1)
        self.activation = nn.ReLU(inplace=True)

        init_weight(self.modules, init_mode)

    def forward(self, x):
        r"""
        Feedforwards a batch of real/fake images and produces a batch of GAN logits.

        Args:
            x (Tensor): A batch of images of shape (N, C, H, W).

        Returns:
            Tensor: A batch of GAN logits of shape (N, 1).
        """
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)

        # GAP (Global Average Pooling)
        h = torch.sum(h, dim=(2, 3))
        output = self.l5(h)

        return output
