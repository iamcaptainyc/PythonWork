
"""
Xception is adapted from https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/xception.py

Ported to pytorch thanks to [tstandley](https://github.com/tstandley/Xception-PyTorch)
@author: tstandley
Adapted by cadene
Creates an Xception Model as defined in:
Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf
This weights ported from the Keras implementation. Achieves the following performance on the validation set:
Loss:0.9173 Prec@1:78.892 Prec@5:94.292
REMEMBER to set your image size to 3x299x299 for both test and validation
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])
The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True, dilation=1):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None

        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=dilation, dilation=dilation, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=dilation,dilation=dilation,bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=dilation,dilation=dilation,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x+=skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, num_classes=1000, downsample_factor=8):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()

        self.num_classes = num_classes
        self.dilation = 1
        if downsample_factor==8:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, True, True]
        else:
            replace_stride_with_dilation = [False, False, False, True]

        self.conv1 = nn.Conv2d(3, 32, 3,2, 0, bias=False) # 1 / 2
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        #do relu here

        self.block1=self._make_block(64,128,2,2,start_with_relu=False,grow_first=True, dilate=replace_stride_with_dilation[0]) # 1 / 4
        self.block2=self._make_block(128,256,2,2,start_with_relu=True,grow_first=True, dilate=replace_stride_with_dilation[1]) # 1 / 8
        self.block3=self._make_block(256,728,2,2,start_with_relu=True,grow_first=True, dilate=replace_stride_with_dilation[2]) # 1 / 16

        self.block4=self._make_block(728,728,3,1,start_with_relu=True,grow_first=True, dilate=replace_stride_with_dilation[2])
        self.block5=self._make_block(728,728,3,1,start_with_relu=True,grow_first=True, dilate=replace_stride_with_dilation[2])
        self.block6=self._make_block(728,728,3,1,start_with_relu=True,grow_first=True, dilate=replace_stride_with_dilation[2])
        self.block7=self._make_block(728,728,3,1,start_with_relu=True,grow_first=True, dilate=replace_stride_with_dilation[2])

        self.block8=self._make_block(728,728,3,1,start_with_relu=True,grow_first=True, dilate=replace_stride_with_dilation[2])
        self.block9=self._make_block(728,728,3,1,start_with_relu=True,grow_first=True, dilate=replace_stride_with_dilation[2])
        self.block10=self._make_block(728,728,3,1,start_with_relu=True,grow_first=True, dilate=replace_stride_with_dilation[2])
        self.block11=self._make_block(728,728,3,1,start_with_relu=True,grow_first=True, dilate=replace_stride_with_dilation[2])

        self.block12=self._make_block(728,1024,2,2,start_with_relu=True,grow_first=False, dilate=replace_stride_with_dilation[3]) # 1 / 32

        self.conv3 = SeparableConv2d(1024,1536,3,1,1, dilation=self.dilation)
        self.bn3 = nn.BatchNorm2d(1536)
        self.relu3 = nn.ReLU(inplace=True)

        #do relu here
        self.conv4 = SeparableConv2d(1536,2048,3,1,1, dilation=self.dilation)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)

        # #------- init weights --------
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        # #-----------------------------

    def _make_block(self, in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True, dilate=False):
        if dilate:
            self.dilation *= strides
            strides = 1
        return Block(in_filters,out_filters,reps,strides,start_with_relu=start_with_relu,grow_first=grow_first, dilation=self.dilation)

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        low_level_feature = self.block1(x)
        x = low_level_feature
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        return low_level_feature, x

    def logits(self, features):
        x = nn.ReLU(inplace=True)(features)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        low_level_feature, x = self.features(input)
        # x = self.logits(x)
        return low_level_feature, x


def _xception(**kwargs):
    print(kwargs)
    url = kwargs.pop('url', -1)
    
    model = Xception(**kwargs)
    
    if url != -1:
        weight_dict=torch.hub.load_state_dict_from_url(url, progress=True)
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
        # 更新现有模型的权重字典
        model_dict.update(state_dict)
        # 载入更新后的权重字典
        model.load_state_dict(model_dict)
    return model


def xception(**kwargs):
    pretrained = kwargs.pop('pretrained', -1)
    if pretrained != -1 and pretrained:
        kwargs['url'] = 'http://data.lip6.fr/cadene/pretrainedmodels/xception-43020ad28.pth'
    
    model = _xception(**kwargs)
    return model

print('xception.py')