# https://github.com/rosinality/stylegan2-pytorch/blob/master/model.py
import math
import random
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from utils import splat_features_from_scores
from .op import FusedLeakyReLU, conv2d_gradfix

__all__ = ["EmptierGAN"]

from .stylegan import PixelNorm, EqualLinear, ConstantInput, Blur, NoiseInjection, Upsample


@dataclass(eq=False)
class EmptierGAN(nn.Module):
    input_nc: int = 3
    output_nc: int = 3
    ngf: int = 64
    n_downsampling: int = 3
    n_blocks: int = 7
    padding_type: str = 'reflect'

    def __post_init__(self):
        super().__init__()
        norm_layer = nn.BatchNorm2d
        assert (self.n_blocks >= 0)
        activation = nn.ReLU(True)
        norm_layer = nn.BatchNorm2d
        model = [nn.ReflectionPad2d(3), nn.Conv2d(self.input_nc, self.ngf, kernel_size=7, padding=0), norm_layer(self.ngf),
                 activation]
        ### downsample
        for i in range(self.n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(self.ngf * mult, self.ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(self.ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2 ** self.n_downsampling
        for i in range(self.n_blocks):
            model += [
                ResnetBlock(self.ngf * mult, padding_type=self.padding_type, activation=activation, norm_layer=norm_layer)]

        ### upsample
        for i in range(self.n_downsampling):
            #print(['loop', i])
            mult = 2 ** (self.n_downsampling - i)
            model += [nn.ConvTranspose2d(self.ngf * mult, int(self.ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                      norm_layer(int(self.ngf * mult / 2)), activation]
            # if i == 0 or i == 1:
            #    model += [Self_Attn(int(ngf * mult / 2), 'relu')]
            #    print(['append', i, int(ngf * mult / 2)])
        model += [nn.ReflectionPad2d(3), nn.Conv2d(self.ngf, self.output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self,
            full_size_input):
        return self.model(full_size_input)



    # Define a resnet block

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