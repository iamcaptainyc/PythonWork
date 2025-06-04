import math
import logging
from functools import partial
from collections import OrderedDict
from copy import Error, deepcopy
from re import S
from numpy.lib.arraypad import pad
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple, trunc_normal_
import torch.fft
from torch.nn.modules.container import Sequential


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class SpectralGatingNetwork(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C)

        x = x.to(torch.float32)

        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')

        x = x.reshape(B, N, C)

        return x

class Block(nn.Module):

    def __init__(self, dim, mlp_ratio=4., h=14, w=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.filter = SpectralGatingNetwork(dim, h=h, w=w)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        x = x + self.mlp(self.norm2(self.filter(self.norm1(x))))
        return x

class Block_attention(nn.Module):

    def __init__(self, dim, mlp_ratio=4.):
        super().__init__()
        num_heads= 4 # 4 for tiny, 6 for small and 12 for base
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)
        self.attn = Attention(dim,  num_heads=num_heads)
 
    def forward(self, x):
        # x = x + self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
         

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        # print(x.shape)
        return x

class PoolPatchEmbed(nn.Module):
    def __init__(self, patch, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch=patch
        self.proj=nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(patch,patch)),
                                nn.Conv2d(in_chans, embed_dim, 1))

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1,2)
        return x

class TransUnet(nn.Module):
    
    def __init__(self, **kwargs):
        super().__init__()
        
        self.args=kwargs['args']
        self.resolution=kwargs['resolution']
        self.downsample_factor= self.args.downsample_factor
        self.module_list = self.args.module_list
        self.backbone= self.args.backbone
        self.num_classes= self.args.num_classes
        self.rsd=[self.downsample_factor<(2**i) for i in range(3,6)]
        if self.backbone == 'rs50':
            self.encoder = resnet50(pretrained=self.args.pretrained, replace_stride_with_dilation=self.rsd)
        elif self.backbone == 'rs101':
            self.encoder = resnet101(pretrained=self.args.pretrained, replace_stride_with_dilation=self.rsd)
        elif self.backbone == 'rs152':
            self.encoder = resnet152(pretrained=self.args.pretrained, replace_stride_with_dilation=self.rsd)
        else:
            assert f'{self.backbone} does not exists!'
        
        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu)#x/2
        self.conv1_out_c=64
        
        self.conv2=nn.Sequential(self.encoder.maxpool,
                                 self.encoder.layer1)#x/4
        self.conv2_out_c=256
        
        self.conv3=nn.Sequential(self.encoder.layer2)#x/8
        self.conv3_out_c=512
        
        self.patch_size=4  # num_features for consistency with other models
        self.feature_size=self.resolution//8
        self.patch=self.feature_size//self.patch_size
        embed_dim=self.patch_size*self.patch_size*3
        self.patch_embed = PatchEmbed(img_size=self.feature_size, in_chans=self.conv3_out_c, patch_size=self.patch_size, embed_dim=embed_dim)
        num_patches = self.patch*self.patch
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        h = self.patch
        w = h // 2 + 1
        
        alpha=2
        depth=6
        self.blocks = nn.ModuleList()
        for i in range(depth):
            if i<alpha:
                layer = Block(dim=embed_dim, h=h, w=w)
                self.blocks.append(layer)
            else:
                layer = Block_attention(dim=embed_dim)
                self.blocks.append(layer) 
        
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.conv = nn.Conv2d(embed_dim, self.conv3_out_c, 1)
        
        # self.dec5 = DecoderBlock(self.conv5_out_c, self.conv4_out_c, is_up=self.rsd[2]==False)
        # self.dec4 = DecoderBlock(self.conv4_out_c, self.conv3_out_c, is_up=self.rsd[1]==False)
        self.dec3 = DecoderBlock(self.conv3_out_c, self.conv2_out_c)
        self.dec2 = DecoderBlock(self.conv2_out_c, self.conv1_out_c)
        
        if self.args.separate_form:
            self.dec_separate = nn.ModuleList([DecoderBlock(self.conv1_out_c, self.conv1_out_c//2, skip_c=3) for i in self.args.separate_form])
            self.final_separate = nn.ModuleList([nn.Conv2d(self.conv1_out_c//2, len(i), kernel_size=1) for i in self.args.separate_form])
        else:
            self.dec1 = DecoderBlock(self.conv1_out_c, self.conv1_out_c//2, skip_c=3)
            self.final = nn.Conv2d(self.conv1_out_c//2, self.num_classes, kernel_size=1)
        

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        # print(x.shape)
        # print(self.pos_embed.shape)
        x = x + self.pos_embed

        
        # print(x.shape)
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        B, N, C = x.shape
        a = b = int(math.sqrt(N))
        x = x.view(B,a,b,C).permute(0,3,1,2)
        x = F.interpolate(x, scale_factor=self.patch)
        return x

    def forward(self, x):
        conv1 = self.conv1(x)#x/2
        conv2 = self.conv2(conv1)#x/4
        conv3 = self.conv3(conv2)#x/8
        # conv4 = self.conv4(conv3)#x/16
        # conv5 = self.conv5(conv4)#x/32
        
        center = self.conv(self.forward_features(conv3))
        
        # dec5 = self.dec5(center, conv4)#x/16
        # dec4 = self.dec4(dec5, conv3)#x/8
        dec3 = self.dec3(conv3, conv2)#x/4
        dec2 = self.dec2(dec3, conv1)#x/2
        
        if self.args.separate_form:
            out=[]
            for i,m in enumerate(self.dec_separate):
                out.append(self.final_separate[i](m(dec2, x)))
            out = torch.cat((out),dim=1)
            return out
        
        dec1 = self.dec1(dec2, x)

        return self.final(dec1)
        return x

print('TransUnet.py')