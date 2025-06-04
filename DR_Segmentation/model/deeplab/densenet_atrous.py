from collections import OrderedDict
from functools import partial
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
NOKEY=-1

# from model.modules import *


class _DenseAtrousLayer(nn.Module):
    def __init__(
        self, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float, memory_efficient: bool = False, dilation: int = 1 
    ) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)

        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)

        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input: Tensor) -> Tensor:  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _DenseAtrousBlock(nn.ModuleDict):
    _version = 2

    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
        memory_efficient: bool = False,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        for i in range(num_layers):
            layer = _DenseAtrousLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
                dilation=dilation
            )
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _AtrousTransition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int, dilation: int = 1) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        if dilation == 1:
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)#减半尺寸


class DenseNetAtrous(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """

    def __init__(
        self,
        growth_rate: int = 32,
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0,
        num_classes: int = 1000,
        memory_efficient: bool = False,
        module_list = None,
        downsample_factor=8,
    ) -> None:

        super().__init__()
        
        # self.module_list = module_list
        # self.attention = module_list['attention']
        
        #x = [3,512,512]
        # First convolution
        self.low_features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),#x/2
                    ("norm0", nn.BatchNorm2d(num_init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )
        self.features = nn.Sequential()
            
        self.cur_downsample_factor=2
        self.dilation=1
        # Each denseblock
        num_features = num_init_features #初始通道数
        for i, num_layers in enumerate(block_config):
            if self.cur_downsample_factor == downsample_factor:
                self.dilation*=2
            else:
                self.dilation=1
                self.cur_downsample_factor*=2
            block = _DenseAtrousBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
                dilation=self.dilation
            )
            if i==0:
                self.low_features.add_module("denseblock%d" % (i + 1), block)
            else:
                self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _AtrousTransition(num_input_features=num_features, num_output_features=num_features // 2, dilation=self.dilation)#减半
                if i==0:
                    self.low_features.add_module("transition%d" % (i + 1), trans)
                    self.low_channels=num_features // 2
                else:
                    self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.output_channels=num_features
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))
        
        print(f'num_features:{num_features}')
        # self.mymodules=get_modules(self.module_list, num_features, num_classes=num_classes)

        # Linear layer
        if isinstance(num_classes, list):
            self.classifier=nn.ModuleList([nn.Linear(num_features, n) for n in num_classes])
        else:
            self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        low_level_feature=self.low_features(x)
        features = self.features(low_level_feature)
        
        # for i in self.mymodules:
        #     features = i(features)
        
        # out = F.relu(features, inplace=True)
        # out = F.adaptive_avg_pool2d(out, (1, 1))
        # out = torch.flatten(out, 1)
        # if isinstance(self.classifier, nn.ModuleList):
        #     x = [classifier(out) for classifier in self.classifier]
        # else:
        #     x = self.classifier(out)
        return low_level_feature, x




def _densenet_atrous(
    growth_rate: int,
    block_config: Tuple[int, int, int, int],
    num_init_features: int,
    **kwargs: Any,
) -> DenseNetAtrous:
    print(kwargs)
    url = kwargs.pop('url', NOKEY)

    model = DenseNetAtrous(growth_rate, block_config, num_init_features, **kwargs)
    
    if url != NOKEY:
        weight_dict=torch.hub.load_state_dict_from_url(url, progress=True)
        del_key = []
        for key, _ in weight_dict.items():  # 遍历预训练权重的有序字典
            if "classifier" in key:  # 如果key中包含'classifier'这个字段
                del_key.append(key)

        for key in del_key:  # 遍历要删除字段的list
            del weight_dict[key]
        # 抽出现有模型中的K,V
        model_dict=model.state_dict()
        # 新建权重字典，并更新
        state_dict={k:v for k,v in weight_dict.items() if k in model_dict.keys()}
        # 更新现有模型的权重字典
        model_dict.update(state_dict)
        # 载入更新后的权重字典
        model.load_state_dict(model_dict)

    return model

DenseNetAtrousWeight={
        'densenet121':'https://download.pytorch.org/models/densenet121-a639ec97.pth',
        'densenet161':'https://download.pytorch.org/models/densenet161-8d451a50.pth',
        'densenet169':'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
        'densenet201':'https://download.pytorch.org/models/densenet201-c1103571.pth'
    }


def densenet_atrous121(**kwargs: Any) -> DenseNetAtrous:
    pretrained = kwargs.pop('pretrained',NOKEY)
    if pretrained != NOKEY and pretrained:
        kwargs['url'] = DenseNetAtrousWeight['densenet121']

    return _densenet_atrous(32, (6, 12, 24, 16), 64, **kwargs)


def densenet_atrous161(**kwargs: Any) -> DenseNetAtrous:
    pretrained = kwargs.pop('pretrained',NOKEY)
    if pretrained != NOKEY and pretrained:
        kwargs['url'] = DenseNetAtrousWeight['densenet161']

    return _densenet_atrous(48, (6, 12, 36, 24), 96, **kwargs)

def densenet_atrous169(**kwargs: Any) -> DenseNetAtrous:
    pretrained = kwargs.pop('pretrained',NOKEY)
    if pretrained != NOKEY and pretrained:
        kwargs['url'] = DenseNetAtrousWeight['densenet169']

    return _densenet_atrous(32, (6, 12, 32, 32), 64, **kwargs)

def densenet_atrous201(**kwargs: Any) -> DenseNetAtrous:
    pretrained = kwargs.pop('pretrained',NOKEY)
    if pretrained != NOKEY and pretrained:
        kwargs['url'] = DenseNetAtrousWeight['densenet201']

    return _densenet_atrous(32, (6, 12, 48, 32), 64, **kwargs)

print('densenet_atrous.py')