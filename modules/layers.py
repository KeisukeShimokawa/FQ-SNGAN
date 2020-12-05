"""
implementation ideas from https://github.com/kwotsin/mimicry/blob/master/torch_mimicry/modules/layers.py
"""
import torch.nn as nn
from torch.nn.utils import spectral_norm


def SNConv2d(*args, **kwargs):
    r"""
    Wrapper for applying Spectral Normalization on official nn.Conv2d 
    """
    return spectral_norm(nn.Conv2d(*args, **kwargs))


def SNLinear(*args, **kwargs):
    r"""
    Wrapper for applying Spectral Normalization on official nn.Linear
    """
    return spectral_norm(nn.Linear(*args, **kwargs))


def SNEmbedding(*args, **kwargs):
    r"""
    Wrapper for applying Spectral Normalization on official nn.Embedding
    """
    return spectral_norm(nn.Embedding(*args, **kwargs))
