from functools import partial
from typing import Any, Callable, List, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch import Tensor
from sympy.strategies import branch
NOKEY=-1

# from model.modules import *


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        
        branch_features = planes * self.expansion
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, module_list = None):
        super(ResNet, self).__init__()
        self.module_list = module_list
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        if isinstance(num_classes, list):
            self.fc=nn.ModuleList([nn.Linear(512 * block.expansion, n) for n in num_classes])
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        if self.module_list:
            self.mymodules=get_modules(self.module_list, 2048, num_classes=num_classes)
        
        self.attmap=self.module_list.get('use_attmap', False)
        if self.attmap:
            self.attmap_low = nn.ModuleList([nn.Sequential(
                                                            nn.Conv2d(1, self.conv1.out_channels//4, 3, 2),
                                                            nn.Conv2d(self.conv1.out_channels//4, self.conv1.out_channels//2, 3, 1),
                                                            nn.BatchNorm2d(self.conv1.out_channels),
                                                            nn.ReLU(inplace=True),
                                                            )
                                            for _ in range(self.attmap)])
            
            self.attmap_high = nn.ModuleList([nn.Sequential(
                                                            nn.Conv2d(self.conv1.out_channels, 1, 1),
                                                            nn.BatchNorm2d(self.conv1.out_channels),
                                                            nn.ReLU(inplace=True),
                                                            nn.AdaptiveAvgPool2d((1, 1))
                                                            )
                                            for _ in range(self.attmap)])
            self.cat_attmap = nn.Linear(512 * block.expansion * self.attmap, 512 * block.expansion)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, attmaps=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        if attmaps:
            refined_attmaps=[]
            low_feature=x
            for i,attmap in enumerate(attmaps):
                attmaps[i] = self.attmap_low[i](attmap) + x
        
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        if self.module_list:
            for i in self.mymodules:
                x = i(x)

        x = self.avgpool(x)
        
        if attmaps:
            high_feature=x
            for i, attmap in enumerate(attmaps):
                refined_attmap=self.attmap_high[i](attmap*high_feature)
                refined_attmaps.append(refined_attmap)
                attmap = refined_attmap+low_feature
                attmap = self.layer1(attmap)
                attmap = self.layer2(attmap)
                attmap = self.layer3(attmap)
                attmap = self.layer4(attmap)
                
                if self.module_list:
                    for i in self.mymodules:
                        attmap = i(attmap)
        
                attmaps[i] = self.avgpool(attmap)
            x = torch.cat(attmaps, dim=1)
            x = self.cat_attmap()
        
        x = torch.flatten(x, 1)
        
        if isinstance(self.fc, nn.ModuleList):
            out = [fc(x) for fc in self.fc]
        else:
            out = self.fc(x)
        
        if attmaps:
            return out, torch.stack(refined_attmaps, dim=0)
        return out


def _resnet(arch, block, layers, **kwargs):
    print(kwargs)
    url = kwargs.pop('url', NOKEY)
    weights = kwargs.pop('weights', NOKEY)
    
    model = ResNet(block, layers, **kwargs)
    
    if url != NOKEY or weights != NOKEY:
        if url != NOKEY:
            print('backbone loading Imagenet weights')
            weight_dict=torch.hub.load_state_dict_from_url(url, progress=True)
        elif weights != NOKEY and weights != None:
            print('backbone loading custome weights')
            weight_dict=torch.load(weights, map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            weight_dict=weight_dict['model_state_dict']
        else:
            return model
        del_key = []
        for key, _ in weight_dict.items():  # 遍历预训练权重的有序字典
            if "fc" in key:  # 如果key中包含'fc'这个字段
                del_key.append(key)

        for key in del_key:  # 遍历要删除字段的list
            del weight_dict[key]
        # 抽出现有模型中的K,V
        model_dict=model.state_dict()
        # 新建权重字典，并更新
        state_dict={k:v for k,v in weight_dict.items() if k in model_dict.keys()}
        # print(model_dict.keys())
        # print(weight_dict.keys())
        # print(state_dict.keys())
        # 更新现有模型的权重字典
        model_dict.update(state_dict)
        # 载入更新后的权重字典
        model.load_state_dict(model_dict)
    return model


def resnet50(**kwargs):
    pretrained = kwargs.pop('pretrained',NOKEY)
    if pretrained != NOKEY and pretrained:
        kwargs['url'] = 'https://download.pytorch.org/models/resnet50-0676ba61.pth'
    
    model = _resnet('ResNet-50', Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    pretrained = kwargs.pop('pretrained',NOKEY)
    if pretrained != NOKEY and pretrained:
        kwargs['url'] = 'https://download.pytorch.org/models/resnet101-cd907fc2.pth'
    
    model = _resnet('ResNet-101', Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    pretrained = kwargs.pop('pretrained',NOKEY)
    if pretrained != NOKEY and pretrained:
        kwargs['url'] = 'https://download.pytorch.org/models/resnet152-f82ba261.pth'
    
    model = _resnet('ResNet-152', Bottleneck, [3, 8, 36, 3], **kwargs)
    return model

def get_resnet_stages(downsample_factor, backbone, module_list, args, **kwargs):
    rsd=[downsample_factor<(2**i) for i in range(3,6)]
    if backbone == 'rs50':
        encoder = resnet50(pretrained=args.pretrained, module_list=module_list, weights=args.backbone_weights, replace_stride_with_dilation=rsd)
    elif backbone == 'rs101':
        encoder = resnet101(pretrained=args.pretrained, module_list=module_list, replace_stride_with_dilation=rsd)
    elif backbone == 'rs152':
        encoder = resnet152(pretrained=args.pretrained, module_list=module_list, replace_stride_with_dilation=rsd)
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

    features = [conv1, conv2, conv3, conv4, conv5]
    channels = [64, 256, 512, 1024, 2048]

    return features, channels

print('resnet.py')
