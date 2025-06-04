# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu}


class TranUnetAttention(nn.Module):
    def __init__(self, hidden_channels, num_heads=8, net_type='linear'):
        super(TranUnetAttention, self).__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_channels / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if net_type == 'linear':
            net=Linear
        elif net_type == 'kan':
            net=KANLinear
            
        self.query = net(hidden_channels, self.all_head_size)
        self.key = net(hidden_channels, self.all_head_size)
        self.value = net(hidden_channels, self.all_head_size)

        self.out = net(hidden_channels, hidden_channels)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        return attention_output


class TranUnetMlp(nn.Module):
    def __init__(self, hidden_channels, expand_ratio=4, net_type='linear'):
        super(TranUnetMlp, self).__init__()
        self.net_type=net_type
        if net_type == 'linear':
            net=Linear
        elif net_type == 'kan':
            net=KANLinear
            
        self.fc1 = net(hidden_channels, hidden_channels*expand_ratio)
        self.fc2 = net(hidden_channels*expand_ratio, hidden_channels)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        if self.net_type == 'linear':
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.normal_(self.fc1.bias, std=1e-6)
            nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, patch_size, img_size, downsample_factor, pretrained=True):
        super(Embeddings, self).__init__()
        self.downsample_factor = downsample_factor
        img_size = img_size//downsample_factor
        patch_size = patch_size
        patch = img_size // patch_size       
        n_patches = patch**2
        # print(f'img_size:{img_size}')
        self.img_size=img_size
        self.patch=patch
        self.hidden_channels=patch_size**2 * 3
        backbone='rs50'
        rsd=[downsample_factor<(2**i) for i in range(3,6)]
        if backbone == 'rs50':
            encoder = resnet50(pretrained=pretrained, replace_stride_with_dilation=rsd)
        elif backbone == 'rs101':
            encoder = resnet101(pretrained=pretrained, replace_stride_with_dilation=rsd)
        elif backbone == 'rs152':
            encoder = resnet152(pretrained=pretrained, replace_stride_with_dilation=rsd)
        else:
            assert f'{backbone} does not exists!'
        
        conv1 = nn.Sequential(encoder.conv1,
                                   encoder.bn1,
                                   encoder.relu)#x/2
        
        conv2=nn.Sequential(encoder.maxpool,
                                 encoder.layer1)#x/4
        
        conv3=nn.Sequential(encoder.layer2)#x/8
        
        conv4=nn.Sequential(encoder.layer3)#x/16
        
        conv5=nn.Sequential(encoder.layer4)#x/32
    
        stages = [conv1, conv2, conv3, conv4, conv5]
        self.stages = nn.ModuleList(stages)
        self.in_channels = [0, 64, 256, 512, 1024]
        self.skip_channels = self.in_channels[::-1]
        
        
        self.patch_embeddings = Conv2d(in_channels=2048,
                                       out_channels=self.hidden_channels,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, self.hidden_channels))

        self.dropout = Dropout(0.1)


    def forward(self, x):
        
        features=[]
        for m in self.stages:
            x=m(x)
            features.append(x)
        #feature不需要conv5，conv5不属于skip
        last_conv=features[-1]
        features=features[:-1]
        features=features[::-1]
        x = self.patch_embeddings(last_conv)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        # print(x.shape)
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        # print(x.shape)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module): # W-MSA in the paper
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, net_type='linear', qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.net_type=net_type
        if net_type == 'linear':
            net=Linear
        elif net_type == 'kan':
            net=KANLinear
        
        self.qkv = net(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = net(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C) >>> (B * 32*32, 4*4, 192)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  #AMBIGUOUS X)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads=12, window_size=7, shift_size=3,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, net_type='linear'):
        super().__init__()
        self.dim = dim
        self.input_resolution = (input_resolution,input_resolution)
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, net_type=net_type)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = TranUnetMlp(dim, net_type=net_type)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class TranUnetBlock(nn.Module):
    def __init__(self, hidden_channels, net_type='linear'):
        super(TranUnetBlock, self).__init__()
        self.hidden_size = hidden_channels
        self.attention_norm = LayerNorm(hidden_channels, eps=1e-6)
        self.ffn_norm = LayerNorm(hidden_channels, eps=1e-6)
        self.ffn = TranUnetMlp(hidden_channels, net_type=net_type)
        self.attn = TranUnetAttention(hidden_channels, net_type=net_type)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x


class Encoder(nn.Module):
    def __init__(self, hidden_channels, num_layers=12, net_type='linear', transformer_block='vit', **kwargs):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(hidden_channels, eps=1e-6)
        
        if transformer_block == 'vit':
            for _ in range(num_layers):
                layer = TranUnetBlock(hidden_channels, net_type=net_type)
                self.layer.append(copy.deepcopy(layer))
        elif transformer_block == 'swin':
            for i in range(num_layers):
                layer = SwinTransformerBlock(dim=hidden_channels, input_resolution=kwargs['input_resolution'],
                                     shift_size=0 if (i % 2 == 0) else 3)
                self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        for layer_block in self.layer:
            # print(hidden_states.shape)
            hidden_states = layer_block(hidden_states)
        encoded = self.encoder_norm(hidden_states)
        return encoded

class Transformer(nn.Module):
    def __init__(self, patch_size, img_size, downsample_factor, num_layers, transformer_block='vit', net_type='linear'):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(patch_size=patch_size, img_size=img_size, downsample_factor=downsample_factor)
        self.encoder = Encoder(self.embeddings.hidden_channels, num_layers, net_type=net_type, transformer_block=transformer_block, input_resolution=self.embeddings.patch)
        self.skip_channels=self.embeddings.skip_channels
        self.hidden_channels=self.embeddings.hidden_channels
        self.patch=self.embeddings.patch

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, features


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class TranUnetDecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        # print(f'x:{x.shape}')
        if skip is not None:
            # print(f'skip:{skip.shape}')
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, hidden_channels, skip_channels, upsample_mode):
        super().__init__()
        head_channels = skip_channels[0]
        self.conv_more = Conv2dReLU(
            hidden_channels,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        self.decoder_channels = [1024, 512, 256, 128, 64, 16]

        self.decoder_channels = self.decoder_channels[6-len(skip_channels):]
        in_channels = [head_channels] + list(self.decoder_channels[:-1]) #[1024, 512,256,128,64]
        out_channels = self.decoder_channels
        self.skip_channels =skip_channels #[1024,512,256,64,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch, mode=upsample_mode) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]

        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        x = self.conv_more(hidden_states)
        for i, decoder_block in enumerate(self.blocks):
            if self.skip_channels[i] != 0:
                x = decoder_block(x, skip=features[i])
            else:
                x = decoder_block(x)
        return x
    
class UnetPPDecoderCup(nn.Module):
    def __init__(
        self, hidden_channels, skip_channels, upsample_mode
    ):
        super().__init__()
        head_channels = skip_channels[0]
        self.conv_more = Conv2dReLU(
            hidden_channels,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        self.decoder_channels = [1024, 512, 256, 128, 64, 16]

        self.decoder_channels = self.decoder_channels[6-len(skip_channels):]
        self.in_channels = [head_channels] + list(self.decoder_channels[:-1])#[512,256,128,64]
        self.out_channels = self.decoder_channels
        self.skip_channels = skip_channels#[512,256,64,0]
        self.depth = len(self.in_channels) - 1

        blocks = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(layer_idx + 1):
                if depth_idx == 0:
                    in_ch = self.in_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1)
                    out_ch = self.out_channels[layer_idx]
                    # print(f'x_{depth_idx}_{layer_idx}--in_ch:{in_ch}')
                    # print(f'x_{depth_idx}_{layer_idx}--skip_ch:{skip_ch}')
                    # print(f'x_{depth_idx}_{layer_idx}--out_ch:{out_ch}')
                else:
                    out_ch = self.skip_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (
                        layer_idx + 1 - depth_idx
                    )
                    in_ch = self.skip_channels[layer_idx - 1]
                blocks[f"x_{depth_idx}_{layer_idx}"] = DecoderBlock(
                    in_ch, out_ch, skip_ch, mode=upsample_mode
                )
        blocks[f"x_{0}_{len(self.in_channels)-1}"] = DecoderBlock(
            self.in_channels[-1], self.out_channels[-1], 0, mode='interp'
        )
        self.blocks = nn.ModuleDict(blocks)
        # print(blocks)

    def forward(self, hidden_states, features=None):
        x = self.conv_more(hidden_states)
        features = [x] + features
        # for i,f in enumerate(features):
        #     print(f'f[i]:{f.shape}')
        # start building dense connections
        dense_x = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(self.depth - layer_idx):
                if layer_idx == 0:
                    # print(f'features[{depth_idx}].shape:{features[depth_idx].shape}')
                    # print(f'features[{depth_idx+1}].shape:{features[depth_idx+1].shape}')
                    output = self.blocks[f"x_{depth_idx}_{depth_idx}"](
                        features[depth_idx], features[depth_idx + 1]
                    )
                    dense_x[f"x_{depth_idx}_{depth_idx}"] = output
                else:
                    dense_l_i = depth_idx + layer_idx
                    cat_features = [
                        dense_x[f"x_{idx}_{dense_l_i}"]
                        for idx in range(depth_idx + 1, dense_l_i + 1)
                    ]
                    cat_features = torch.cat(
                        cat_features + [features[dense_l_i + 1]], dim=1
                    )
                    dense_x[f"x_{depth_idx}_{dense_l_i}"] = self.blocks[
                        f"x_{depth_idx}_{dense_l_i}"
                    ](dense_x[f"x_{depth_idx}_{dense_l_i-1}"], cat_features)
        dense_x[f"x_{0}_{self.depth}"] = self.blocks[f"x_{0}_{self.depth}"](
            dense_x[f"x_{0}_{self.depth-1}"]
        )

        last_out = dense_x[f"x_{0}_{self.depth}"]
        return last_out


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, num_classes=4, downsample_factor=8, upsample_mode='interp', transformer_block='vit', net_type='linear', decoder='unet'):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.patch_size= 8 if img_size<=512 else 16
        self.transformer = Transformer(patch_size=self.patch_size, img_size=img_size, downsample_factor=downsample_factor, num_layers=12, transformer_block=transformer_block, net_type=net_type)
        self.downsample_factor=downsample_factor
        if decoder == 'unet':
            self.decoder = DecoderCup(self.transformer.hidden_channels, self.transformer.skip_channels, upsample_mode)
        elif decoder == 'unetpp':
            self.decoder = UnetPPDecoderCup(self.transformer.hidden_channels, self.transformer.skip_channels, upsample_mode)
        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.decoder_channels[-1],
            out_channels=num_classes,
            kernel_size=3,
        )

    def forward(self, x):
        x, features = self.transformer(x)  # (B, n_patch, hidden)
        # print(f'transformer x:{x.shape}')
        B, n_patch, hidden = x.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        # print(f'before up:{x.shape}')
        x = F.interpolate(x, scale_factor=self.patch_size, mode="nearest")
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits
    
print('transunet.py')