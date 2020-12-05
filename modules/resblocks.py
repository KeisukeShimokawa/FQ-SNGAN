r"""
implementation ideas from https://github.com/kwotsin/mimicry/blob/master/torch_mimicry/modules/layers.py
"""
import torch.nn as nn
import torch.nn.functional as F

from modules.layers import SNConv2d, SNLinear
from modules.batchnorm import ConditionalBatchNorm2d


class GBlock(nn.Module):
    r"""
    Residual block for generator.

    Uses bilinear (rather than nearest) interpolation, and align_corners
    set to False. This is as per how torchvision does upsampling, as seen in:
    https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/_utils.py

    Attributes:
        in_channels (int): The channel size of input feature map.
        out_channels (int): The channel size of output feature map.
        hidden_channels (int): The channel size of intermediate feature maps.
        upsample (bool): If True, upsamples the input feature map.
        num_classes (int): If more than 0, uses conditional batch norm instead.
        spectral_norm (bool): If True, uses spectral norm for convolutional layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=None,
                 upsample=False,
                 num_classes=0,
                 spectral_norm=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels if hidden_channels is not None else out_channels
        self.learnable_sc = (in_channels != out_channels) or upsample
        self.upsample = upsample
        self.num_classes = num_classes
        self.spectral_norm = spectral_norm

        if self.spectral_norm:
            self.c1 = SNConv2d(self.in_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1)
            self.c2 = SNConv2d(self.hidden_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        else:
            self.c1 = nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1)
            self.c2 = nn.Conv2d(self.hidden_channels, self.out_channels, kernel_size=3, stride=1, padding=1)

        if self.num_classes == 0:
            self.b1 = nn.BatchNorm2d(self.in_channels)
            self.b2 = nn.BatchNorm2d(self.hidden_channels)
        else:
            self.b1 = ConditionalBatchNorm2d(self.in_channels, self.num_classes)
            self.b2 = ConditionalBatchNorm2d(self.hidden_channels, self.num_classes)

        if self.learnable_sc:
            if self.spectral_norm:
                self.c_sc = SNConv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
            else:
                self.c_sc = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)

        self.activation = nn.ReLU(inplace=True)

    def shortcut(self, x):
        r"""
        Helper function for feedforwarding through shortcut layers.
        """
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        if self.learnable_sc:
            x = self.c_sc(x)

        return x

    def residual(self, x, y):
        r"""
        Helper function for feedforwarding through main layers.
        """
        h = x
        if self.num_classes == 0:
            h = self.b1(h)
        else:
            h = self.b1(h, y)
        h = self.activation(h)

        if self.upsample:
            h = F.interpolate(h, scale_factor=2, mode='bilinear', align_corners=False)

        h = self.c1(h)

        if self.num_classes == 0:
            h = self.b2(h)
        else:
            h = self.b2(h, y)
        h = self.activation(h)

        h = self.c2(h)

        return h


    def forward(self, x, y=None):
        r"""
        Residual block feedforward function.

        Args:
            x (Tensor): Input Feature map after convolution of previous block.
            y (Tensor): Input class labels.

        Returns:
            Tensor: Output feature map
        """
        return self.residual(x, y) + self.shortcut(x)
        

class DBlock(nn.Module):
    r"""
    Residual block for discriminator.

    Attributes:
        in_channels (int): The channel size of input feature map.
        out_channels (int): The channel size of output feature map.
        hidden_channels (int): The channel size of intermediate feature maps.
        downsample (bool): If True, downsamples the input feature map.
        spectral_norm (bool): If True, uses spectral norm for convolutional layers.        
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=None,
                 downsample=False,
                 spectral_norm=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels if hidden_channels is not None else in_channels
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        self.spectral_norm = spectral_norm

        if self.spectral_norm:
            self.c1 = SNConv2d(self.in_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1)
            self.c2 = SNConv2d(self.hidden_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        else:
            self.c1 = nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1)
            self.c2 = nn.Conv2d(self.hidden_channels, self.out_channels, kernel_size=3, stride=1, padding=1)

        if self.learnable_sc:
            if self.spectral_norm:
                self.c_sc = SNConv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
            else:
                self.c_sc = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)

        self.activation = nn.ReLU(inplace=True)
            
    def shortcut(self, x):
        r"""
        Helper function for feedforwarding through shortcut layers.
        """
        if self.learnable_sc:
            x = self.c_sc(x)

        if self.downsample:
            x = F.avg_pool2d(x, kernel_size=2)

        return x

    def residual(self, x):
        r"""
        Helper function for feedforwarding through main layers.
        """
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = F.avg_pool2d(h, kernel_size=2)

        return h

    def forward(self, x):
        r"""
        Args:
            x (Tensor): Input Feature map after convolution of previous block.

        Returns:
            Tensor: Output feature map
        """
        return self.residual(x) + self.shortcut(x)


class DBlockOptimized(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 spectral_norm=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spectral_norm = spectral_norm

        if self.spectral_norm:
            self.c1 = SNConv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
            self.c2 = SNConv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
            self.c_sc = SNConv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.c1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
            self.c2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
            self.c_sc = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)

        self.activation = nn.ReLU(inplace=True)

    def shortcut(self, x):
        r"""
        Helper function for feedforwarding through shortcut layers.
        """
        return self.c_sc(F.avg_pool2d(x, kernel_size=2))

    def residual(self, x):
        r"""
        Helper function for feedforwarding through main layers.
        """
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = self.activation(h)
        h = F.avg_pool2d(h, kernel_size=2)

        return h

    def forward(self, x):
        r"""
        Args:
            x (Tensor): Input Feature map after convolution of previous block.

        Returns:
            Tensor: Output feature map
        """
        return self.residual(x) + self.shortcut(x)
