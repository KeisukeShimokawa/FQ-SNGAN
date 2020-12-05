"""
implementation ideas from https://github.com/kwotsin/mimicry/blob/master/torch_mimicry/modules/layers.py
"""
import torch.nn as nn


class ConditionalBatchNorm2d(nn.Module):
    r"""
    Conditional Batch Norm as implemented in
    https://github.com/pytorch/pytorch/issues/8985
    
    Attributes:
        num_features (int): Size of feature map for batch norm.
        num_classes (int): Determines size of embedding layer to condition BN.
    """
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)   # initialize scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()            # initialize bias  at 0

    def forward(self, x, y):
        r"""
        FeedForward for conditional batch norm
        
        Args:
            x (Tensor): Input feature map.
            y (Tensor): Input class labels for embedding.
        
        Returns:
            Tensor: Output feature map.
        """
        out = self.bn(x)
        # divide into 2 chunks, [B, num_features * 2] --> [B, num_features], [B, num_features]
        gamma, beta = self.embed(y).chunk(2, dim=1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + \
               beta.view(-1, self.num_features, 1, 1)

        return out
