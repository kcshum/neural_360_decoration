# https://github.com/rosinality/stylegan2-pytorch/blob/master/model.py
import random
from dataclasses import dataclass
from typing import Optional, Dict

import torch
from einops import rearrange
from torch import nn, Tensor

__all__ = ["twoDLayoutEncoder"]

from models.networks.stylegan import StyleMLP, pixel_norm
from utils import derange_tensor


@dataclass(eq=False)
class twoDLayoutEncoder(nn.Module):
    input_nc: int = 3
    output_nc: int = 3
    ngf: int = 4
    n_downsampling: int = 3
    n_blocks: int = 3
    padding_type: str = 'reflect'

    max_downsample_dim: int = 32

    size: int = 256
    rect_scaler: int = 1
    force_square: bool = False


    # bg options
    use_bg_before_mlp: bool = False
    # parameters
    twoD_noise_dim: int = 3
    indicator_dim: int = 1

    feature_dim: int = 512
    style_dim: int = 512
    # MLP options
    mlp_n_layers: int = 1
    mlp_trunk_n_layers: int = 4
    mlp_hidden_dim: int = 1024
    n_features_max: int = 5
    norm_features: bool = False
    # Transformer options
    spatial_style: bool = False
    # Training options
    mlp_lr_mul: float = 0.01
    shuffle_features: bool = False
    p_swap_style: float = 0.0
    feature_jitter_xy: float = 0.0  # Legacy, unused
    feature_dropout: float = 0.0

    def __post_init__(self):
        super().__init__()
        norm_layer = nn.BatchNorm2d
        assert (self.n_blocks >= 0)
        activation = nn.ReLU(True)
        norm_layer = nn.BatchNorm2d
        model = [nn.ReflectionPad2d(3), nn.Conv2d(self.input_nc + self.twoD_noise_dim + self.indicator_dim, self.ngf, kernel_size=7, padding=0), norm_layer(self.ngf),
                 activation]
        ### downsample
        for i in range(self.n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(min(self.ngf * mult, self.max_downsample_dim), min(self.ngf * mult * 2, self.max_downsample_dim), kernel_size=3, stride=2, padding=1),
                      norm_layer(min(self.ngf * mult * 2, self.max_downsample_dim)), activation]

        ### resnet blocks
        mult = 2 ** self.n_downsampling
        for i in range(self.n_blocks):
            model += [
                ResnetBlock(min(self.ngf * mult, self.max_downsample_dim), padding_type=self.padding_type, activation=activation, norm_layer=norm_layer)]

        self.twoDEncoder = nn.Sequential(*model)

        if self.force_square:
            self.downsampler = nn.Upsample(size=[self.size, self.size])
            self.rect_scaler = 1

        size = self.size // (2 ** self.n_downsampling)
        first_dim = min(self.ngf * mult, self.max_downsample_dim) * size * size * self.rect_scaler

        self.adapter = nn.Sequential(
            nn.Flatten(),
            nn.Linear(first_dim, self.mlp_hidden_dim)
        )

        # {x_i, y_i, feature_i, covariance_i}, bg feature, and cluster sizes
        maybe_style_dim = int(self.spatial_style) * self.style_dim
        ndim = (self.feature_dim + maybe_style_dim + 2 + 4 + 1) * self.n_features_max + \
               (maybe_style_dim + self.feature_dim + 1)

        self.mlp = StyleMLP(self.mlp_n_layers, self.mlp_hidden_dim, self.mlp_lr_mul, first_dim=self.mlp_hidden_dim,
                            last_dim=ndim, last_relu=False)
        print(first_dim, self.mlp_hidden_dim, ndim)


    def forward(self, inp: Tensor, twoD_z: Tensor, n_features: int) -> Optional[Dict[str, Tensor]]:
        """
        Args:
            noise: [size x size towD_noise_dim]
            mlp_idx: which IDX to start running MLP from, useful for truncation
            n_features: int num features to output
        Returns: three tensors x coordinates [N x M], y coordinates [N x M], features [N x M x feature_dim]
        """
        inp_and_noise = torch.cat((inp, twoD_z), 1)
        if self.force_square:
            inp_and_noise = self.downsampler(inp_and_noise)
        out = self.twoDEncoder(inp_and_noise)
        out = self.adapter(out)
        out = self.mlp(out)

        #print("out")
        #print(out.shape)
        sizes, out = out.tensor_split((self.n_features_max + 1,), dim=1)
        #print("sizes", "out")
        #print(sizes.shape, out.shape)

        bg_feat, out = out.tensor_split((self.feature_dim,), dim=1)
        #print("bg_feat", "out")
        #print(bg_feat.shape, out.shape)
        if self.spatial_style:
            bg_style_feat, out = out.tensor_split((self.style_dim,), dim=1)
            #print("bg_style_feat", "out")
            #print(bg_style_feat.shape, out.shape)

        out = rearrange(out, 'n (m d) -> n m d', m=self.n_features_max)
        #print("out")
        #print(out.shape)

        if self.shuffle_features:
            idxs = torch.randperm(self.n_features_max)[:n_features]
        else:
            idxs = torch.arange(n_features)
        out = out[:, idxs]
        sizes = sizes[:, [0] + idxs.add(1).tolist()]


        if self.feature_dropout:
            keep = torch.rand((out.size(1),)) > self.feature_dropout
            if not keep.any():
                keep[0] = True
            out = out[:, keep]
            sizes = sizes[:, [True] + keep.tolist()]
            #print('feature_dropout')
        xy = out[..., :2].sigmoid()  # .mul(self.max_coord)

        ret = {'xs': xy[..., 0], 'ys': xy[..., 1], 'sizes': sizes[:, :n_features + 1], 'covs': out[..., 2:6]}
        '''
        print('ret')
        print('ret:xs:ys:sizes:covs')
        print(ret['xs'].size())
        print(ret['ys'].size())
        print(ret['sizes'].size())
        print(ret['covs'].size())
        '''

        end_dim = self.feature_dim + 6
        features = out[..., 6:end_dim]

        features = torch.cat((bg_feat[:, None], features), 1)
        ret['features'] = features
        # return [xy[..., 0], xy[..., 1], features, covs, sizes[:, :n_features + 1].softmax(-1)]
        if self.spatial_style:
            style_features = out[..., end_dim:]
            style_features = torch.cat((bg_style_feat[:, None], style_features), 1)
            ret['spatial_style'] = style_features
        # ret['covs'] = ret['covs'].detach()
        if self.norm_features:
            for k in ('features', 'spatial_style', 'shape_features'):
                if k in ret:
                    ret[k] = pixel_norm(ret[k])
            #print('norm_features')
        if self.p_swap_style:
            if random.random() <= self.p_swap_style:
                n = random.randint(0, ret['spatial_style'].size(1) - 1)
                shuffle = torch.randperm(ret['spatial_style'].size(1) - 1).add(1)[:n]
                ret['spatial_style'][:, shuffle] = derange_tensor(ret['spatial_style'][:, shuffle])
            #print('p_swap_style')
        return ret



class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(
            self,
            x):

        out = x + self.conv_block(x)
        return out