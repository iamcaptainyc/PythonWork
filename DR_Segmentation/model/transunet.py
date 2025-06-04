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

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu}


class TranUnetAttention(nn.Module):
    def __init__(self, hidden_channels, num_heads=8, net_type='linear'):
        super(TranUnetAttention, self).__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_channels / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if net_type == 'mlp':
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
    def __init__(self, patch_size, img_size, pretrained=True):
        super(Embeddings, self).__init__()
        img_size = img_size//8
        patch_size = patch_size
        patch = img_size // patch_size       
        n_patches = patch**2
        # print(f'img_size:{img_size}')

        self.patch=patch
        self.hidden_channels=patch_size**2 * 3
        self.encoder = resnet50(pretrained=pretrained)
        
        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu)#x/2
        self.conv1_out_c=64
        
        self.conv2=nn.Sequential(self.encoder.maxpool,
                                 self.encoder.layer1)#x/4
        self.conv2_out_c=256
        
        self.conv3=nn.Sequential(self.encoder.layer2)#x/8
        self.conv3_out_c=512
        self.skip_channels=[0, self.conv1_out_c, self.conv2_out_c, self.conv3_out_c]
        self.skip_channels=self.skip_channels[::-1]
        
        in_channels=self.conv3_out_c
        
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=self.hidden_channels,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, self.hidden_channels))

        self.dropout = Dropout(0.1)


    def forward(self, x):
        conv1 = self.conv1(x)#x/2
        conv2 = self.conv2(conv1)#x/4
        conv3 = self.conv3(conv2)#x/8
        
        features=[conv1, conv2, conv3]
        features=features[::-1]
        
        x = self.patch_embeddings(conv3)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        # print(x.shape)
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        # print(x.shape)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


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
    def __init__(self, hidden_channels, num_layers=12, net_type='linear'):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(hidden_channels, eps=1e-6)
        for _ in range(num_layers):
            layer = TranUnetBlock(hidden_channels, net_type=net_type)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        for layer_block in self.layer:
            # print(hidden_states.shape)
            hidden_states = layer_block(hidden_states)
        encoded = self.encoder_norm(hidden_states)
        return encoded


class Transformer(nn.Module):
    def __init__(self, patch_size, img_size, num_layers, net_type='linear'):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(patch_size=patch_size, img_size=img_size)
        self.encoder = Encoder(self.embeddings.hidden_channels, num_layers, net_type=net_type)
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
        self.decoder_channels = [256, 128, 64, 16]
        in_channels = [head_channels] + list(self.decoder_channels[:-1])
        out_channels = self.decoder_channels
        self.skip_channels =skip_channels

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch, mode=upsample_mode) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]

        if upsample_mode == 'freqfusion':
            blocks[-1]=DecoderBlock(in_channels[-1], out_channels[-1], skip_channels[-1], mode='interp')
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        x = self.conv_more(hidden_states)
        for i, decoder_block in enumerate(self.blocks):
            if self.skip_channels[i] != 0:
                x = decoder_block(x, skip=features[i])
            else:
                x = decoder_block(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, num_classes=4, upsample_mode='interp', net_type='linear'):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.patch_size= 8 if img_size<=512 else 16
        self.transformer = Transformer(patch_size=self.patch_size, img_size=img_size, num_layers=12, net_type=net_type)
        self.decoder = DecoderCup(self.transformer.hidden_channels, self.transformer.skip_channels, upsample_mode)
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
        x = F.interpolate(x, scale_factor=self.patch_size//2, mode="nearest")
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits
    
print('transunet.py')