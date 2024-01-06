# https://github.com/rosinality/stylegan2-pytorch/blob/master/model.py
import random
from dataclasses import dataclass
from typing import Optional, Dict


import torch
from einops import rearrange
from torch import nn, Tensor

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor

__all__ = ["bgEncoder"]

from models.networks.stylegan import StyleMLP, pixel_norm
from utils import derange_tensor

class originEncoder(nn.Module):
    def forward(self, background: Tensor):
        return background

class resnet(nn.Module):
    def __init__(self, feature_dim, style_dim):
        super().__init__()
        encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        for child in encoder.children():
            for param in child.parameters():
                param.requires_grad = False

        return_nodes = {
            # node_name: user-specified key for output dict
            'layer1': 'layer1',
            'layer2': 'layer2',
            'layer3': 'layer3',
            'layer4': 'layer4',
        }
        self.feature_extractor = create_feature_extractor(encoder, return_nodes=return_nodes)
        self.adapter = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048*8*8, feature_dim + style_dim)
        )

    def forward(self, background: Tensor):
        out = self.feature_extractor(background)
        out = out['layer4']

        out = self.adapter(out)
        return out

@dataclass(eq=False)
class bgEncoder(nn.Module):
    #model type
    pretrained_model: str = 'origin'
    #para
    feature_dim: int = 512
    style_dim: int = 512
    # Training options
    mlp_lr_mul: float = 0.01
    shuffle_features: bool = False
    p_swap_style: float = 0.0
    feature_jitter_xy: float = 0.0  # Legacy, unused
    feature_dropout: float = 0.0

    def __post_init__(self):
        super().__init__()
        if self.feature_jitter_xy:
            print('Warning! This parameter is here only to support loading of old checkpoints, and does not function. '
                  'Unless you are loading a model that has this value set, it should not be used. To control jitter, '
                  'set model.feature_jitter_xy directly.')
        # {x_i, y_i, feature_i, covariance_i}, bg feature, and cluster sizes
        #maybe_style_dim = int(self.spatial_style) * self.style_dim
        #ndim = maybe_style_dim + self.feature_dim + 1
        #resnet_encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        #self.encoder =
        #self.adapter = nn.Linear(in_features=512, out_features=ndim)
        if self.pretrained_model == 'origin':
            self.encoder = originEncoder()
        elif self.pretrained_model == 'resnet50':
            self.encoder = resnet(self.feature_dim, self.style_dim)

    def forward(self, background: Tensor,
                mlp_idx: Optional[int] = None) -> Optional[Dict[str, Tensor]]:
        """
        Args:
            backgound:
        Returns:
        """
        out = self.encoder(background)
        return {'bg': out}

