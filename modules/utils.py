r"""
implementation from https://github.com/POSTECH-CVLab/PyTorch-StudioGAN/blob/7b5ef8dae5dc78d28322e7c05f6ce8b13c380573/src/utils/model_ops.py
"""
import torch
import torch.nn as nn
from torch.nn import init


def init_weight(modules, mode):

    assert mode in ["ortho", "N02", "glorot", "xavier"], 'set mode in ["ortho", "N02", "glorot", "xavier"]'

    for module in modules():
        if (isinstance(module, nn.Conv2d) 
            or isinstance(module, nn.Linear)
            or isinstance(module, nn.ConvTranspose2d)):
            if mode == "ortho":
                init.orthogonal_(module.weight)
            
            if mode == "N02":
                init.normal_(module.weight, 0, 0.02)

            if mode in ["glorot", "xavier"]:
                init.xavier_uniform_(module.weight)

            if module.bias is not None:
                module.bias.data.zero_()
        else:
            pass
