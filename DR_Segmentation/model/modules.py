import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch import Tensor
import math
import cv2
import torch_dct as dct
import einops
# from model.dca import *
# from sklearn.cross_decomposition._pls import CCA
NOKEY=-1

c2wh = dict([(64,56), (128,28), (256,14) ,(512,7)])
my_c2wh = dict([(512,16),(256,32)])
freq_method='top16'
end_modules=['mccsa']

def channel2wh(channels):
    channels=channels//4
    wh=my_c2wh.get(channels, -1)
    if wh == -1:
        if channels < 256:
            wh=64
        else:
            wh=16
    return wh

def get_modules(module_dict, out_c, **kwargs):
    default_values={'fm':'top16'}
    mymodules=nn.ModuleList()
    module_list=list(module_dict.values())
    for i in module_list:
        m=None
        if i == 'aspp':
            m=ASPPModule(out_c)
        elif i == 'biaspp':
            m=ASPPModule(out_c,bi=True)
        elif i == 'sa':
            m=sa_layer(out_c)
        elif i == 'cbam':
            m=nn.Sequential(
                ChannelAttention(out_c),
                SpatialAttention(),
            )
        elif i == 'myffs':
            m=nn.Sequential(
                DFCAM(out_c, channel2wh(out_c), channel2wh(out_c), freq_size=8, freq_sel_method = module_dict.get('freq_method', 'dct16'), fc='conv'),
                FSpatialAttention(transform='dct'),
            )
        
            
        if m!=None: mymodules.append(m)
    return mymodules

def transformTo(x, transform='fft'):
    if transform == 'dct':
        x_np=x.detach().cpu().numpy()
#         for b in range(x_np.shape[0]):
#             for c in range(x_np.shape[1]):
#                 x_np[b,c]=cv2.dct(x_np[b,c])
        x=dct.dct_2d(x)
        x=torch.from_numpy(x_np).to(x.device)
        return x
    elif transform == 'fft':
        return torch.real(torch.fft.fft2(x, dim=(-2, -1),norm='forward'))

'''
#ASPP
'''
   
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPPModule(nn.Module):
    def __init__(self, in_channels, atrous_rates=[6, 12, 18], bi=False):
        super(ASPPModule, self).__init__()
        self.bi=bi
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
#         print('aspp')
        res = []
        for conv in self.convs:
            res.append(conv(x))
            
        if self.bi:
            mid=res[1]+res[2]
            res[3]=mid+res[3]
            res[2]=mid+res[3]
            res[1]=res[1]+res[2]
        
        res = torch.cat(res, dim=1)
        return self.project(res)


'''
Shuffle Attention
'''

class sa_layer(nn.Module):
    """Constructs a Channel Spatial Group module.

    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, groups=64):
        super(sa_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
#         print('sa')
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out
    

    
'''
CBAM
'''

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
#         print('cabm')
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        out = self.sigmoid(out)*x
        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return self.sigmoid(out)*x
    
class FChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(FChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        f_x=transformTo(x)
        f_avg_out = self.fc(self.avg_pool(f_x))
        f_max_out = self.fc(self.max_pool(f_x))
        f_out=f_avg_out + f_max_out
        f_out=self.sigmoid(f_out)*f_x
        return f_out
    
class FSpatialAttention(nn.Module):
    def __init__(self,transform='fft', kernel_size=7):
        super(FSpatialAttention, self).__init__()

        self.transform=transform
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        f_x=transformTo(x, self.transform)
        f_avg_out = torch.mean(f_x, dim=1, keepdim=True)
        f_max_out, _ = torch.max(f_x, dim=1, keepdim=True)
        f_out = torch.cat([f_avg_out, f_max_out], dim=1)
        
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        
        out = out*f_out
        out = self.conv1(out)
        
        return self.sigmoid(out)*x

'''
FcaNet
'''
   
def get_freq_indices(method):
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32',
                      'dct2','dct4','dct8','dct16','dct32',
                      'idr2','idr4','idr8','idr16','idr32',]
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    elif 'dct' in method:
        all_bot_indices_x = [0, 4, 1, 3, 2, 0, 6, 0, 5, 2, 4, 3, 7, 6, 2, 2, 6, 2, 6, 6, 1, 5, 5, 6, 7, 5, 0, 4, 3, 4, 0, 5]
        all_bot_indices_y = [0, 2, 4, 7, 4, 6, 0, 5, 1, 6, 4, 0, 6, 5, 3, 7, 4, 1, 3, 6, 0, 3, 4, 1, 3, 7, 7, 5, 6, 1, 1, 6]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    elif 'idr' in method:
        all_bot_indices_x = [0, 0, 1, 0, 2, 3, 4, 4, 6, 6, 0, 0, 2, 3, 7, 0, 1, 6, 6, 6, 7, 7, 0, 1, 1, 2, 2, 4, 5, 5, 5, 6]
        all_bot_indices_y = [0, 1, 6, 3, 0, 0, 1, 6, 0, 1, 4, 7, 7, 6, 2, 2, 3, 3, 4, 6, 4, 6, 5, 0, 2, 3, 6, 5, 0, 2, 7, 5]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction = 16, freq_size=7, freq_sel_method = 'top16'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // freq_size) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // freq_size) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n,c,h,w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered. 
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)
    
class DCTSpectralLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction = 16, freq_size=8, mapper=[0,0]):
        super(DCTSpectralLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = [mapper[0]],[mapper[1]]
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // freq_size) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // freq_size) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n,c,h,w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered. 
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()
        
        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        
        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        
        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight

        result = torch.sum(x, dim=[2,3])
        return result

    def build_filter(self, pos, freq, POS): #即基础函数
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS) 
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)
    
    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i+1)*c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)
                    #dct_filter即得到每个channel对应的基础函数
                        
        return dct_filter
    
'''
    FCAM
'''
class DFCAM(nn.Module):
    def __init__(self, in_planes, dct_h, dct_w, reduction = 16,freq_size=8, freq_sel_method = 'top16', fc='conv', transform=None):
        super(DFCAM, self).__init__()
        
        self.transform=transform
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // freq_size) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // freq_size) for temp_y in mapper_y]

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, in_planes)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc_method = fc
        
        if fc == 'linear':
            self.fc = nn.Sequential(
                nn.Linear(in_planes, in_planes // reduction, bias=False),#这里用的是Linear，所以输入形状必须是【bs,channels】
                nn.ReLU(inplace=True),
                nn.Linear(in_planes // reduction, in_planes, bias=False),
            )
        elif fc == 'conv':
            self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False))
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        n,c,h,w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered. 
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)

        if self.fc_method == 'conv':
            y=y.view(n,c,1,1)
        y = self.sigmoid(self.fc(y)).view(n, c, 1, 1)
        dct_att = x * y.expand_as(x)
        
        
        if self.transform:
            f_x=transformTo(x, self.transform)
        else:
            f_x=x
        
        if self.fc_method == 'linear':
            f_avg_out = self.fc(self.avg_pool(f_x).squeeze())
            f_max_out = self.fc(self.max_pool(f_x).squeeze())
        else:
            f_avg_out = self.fc(self.avg_pool(f_x))
            f_max_out = self.fc(self.max_pool(f_x))
        f_out=f_avg_out + f_max_out
        f_out=f_out.view(n, c, 1, 1)
        f_out=self.sigmoid(f_out)*f_x
        
        return dct_att + f_out
    
    
'''
Global Transformer block
'''
class ChannelSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelSelfAttention, self).__init__()
        
        self.inter_channels=in_channels//8
        self.conv1 = nn.Conv2d(in_channels, self.inter_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, self.inter_channels, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels, self.inter_channels, kernel_size=3, padding=1, bias=False)
        self.lastconv = nn.Conv2d(self.inter_channels, in_channels, 1, bias=False)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        q=self.gap(self.conv1(x)).flatten(2) # [b,c,1]
        
        k_T=self.conv2(x).flatten(2).permute(0,2,1) # [b,h*w,c]
        
        v=self.conv3(x).flatten(2) # [b,c,h*w]
        
        att = torch.matmul(k_T, q) # [b,h*w,1]
        att = F.softmax(att, dim=1)
        out = torch.matmul(v, att)
        out = out.view(out.shape[0],out.shape[1],1,1)
        out = self.lastconv(out)
        out = out + x
        
        return out
    
class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels, extra_channels=None):
        super(SpatialSelfAttention, self).__init__()
        
        self.inter_channels=in_channels//2
        self.conv1 = nn.Conv2d(in_channels, self.inter_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, self.inter_channels, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels, self.inter_channels, kernel_size=3, padding=1, bias=False)
        self.lastconv = nn.Conv2d(self.inter_channels, in_channels, 1, bias=False)
        
        if extra_channels is not None:
            self.extra_project=nn.Conv2d(extra_channels, in_channels, 1, bias=False)
            

    def forward(self, x, extra=None):
        b, c, h, w = x.shape
        
        q=self.conv1(x).view(b, self.inter_channels, -1)
        
        if extra is not None:
            extra=self.extra_project(extra)
            feature=extra
        else:
            feature=x
            
        k_T=self.conv2(feature).view(b, self.inter_channels, -1).permute(0,2,1)
        v=self.conv3(feature).view(b, self.inter_channels, -1)
        att = torch.matmul(k_T, q)
        att = F.softmax(att, dim=-1)
        
        out = torch.matmul(v, att)
        out = out.view(b, self.inter_channels, h, w)
        
        out = self.lastconv(out)
        out = out + x
        
        return out
    
class RTBlock(nn.Module):
    def __init__(self, in_channels, patch=None):
        super().__init__()
        self.patch=patch
        if self.patch:
            self.patch_avg = nn.ModuleList([nn.AdaptiveAvgPool2d(output_size=(patch, patch)) for _ in range(2)]) #b h w c->b p*p c
        
            self.avg_map = nn.ModuleList([
                    nn.Sequential(nn.Conv2d(in_channels, in_channels, 1),
                                  nn.BatchNorm2d(in_channels), 
                                  nn.ReLU(inplace=True))
                    for _ in range(2)
                ])

            # self.patch_avg = nn.ModuleList([nn.Conv2d(in_channels,in_channels,kernel_size=patch,stride=patch) for _ in range(2)])
            # self.avg_map = nn.ModuleList([nn.Identity() for _ in range(2)])
            
            self.upconv = nn.ModuleList([
                    nn.Conv2d(in_channels, in_channels, 1) for _ in range(2)
                ])
            
        # self.csa=ChannelSelfAttention(in_channels)
        # self.csa_extra=ChannelSelfAttention(in_channels)
        self.ssa=SpatialSelfAttention(in_channels)
        self.ssa_cross=SpatialSelfAttention(in_channels, in_channels)
        
        self.conv = nn.Conv2d(in_channels*2, in_channels, kernel_size=1, bias=False)
        
    def forward(self, x, extra=None):
        raw_size = x.shape[2]
        if self.patch:
            x = self.patch_avg[0](x)
            x = self.avg_map[0](x)
            if extra != None:
                extra = self.patch_avg[1](extra)
                extra = self.avg_map[1](extra)
        
        # if extra != None:
        #     x = self.csa(x)
        #     extra = self.csa_extra(extra)
        # else:
        #     raw_featrues=x
        #     x = self.csa(x)
        #     extra = self.csa_extra(raw_featrues)
        
        self_att = self.ssa(x)
        cross_att = self.ssa_cross(x, extra)
        
        x = self.conv(torch.cat((self_att,cross_att), dim=1))
        
        if self.patch:
            x = self.upsample(x, raw_size)
            extra = self.upsample(extra, raw_size)
            x = self.upconv[0](x)
            extra = self.upconv[1](extra)
            
        return x, extra

    def upsample(self, x, raw_size):
        if raw_size == x.shape[2]:
            return x
        x = F.interpolate(x, scale_factor=raw_size//x.shape[2], mode="nearest")
        return x
    
class DualCrossAttentionBlock(nn.Module):
    def __init__(self, hidden_channels, patch=None, net_type='linear'):
        super().__init__()
        self.hidden_size = hidden_channels
        self.attention_norm = nn.ModuleList([LayerNorm(hidden_channels, eps=1e-6) for _ in range(2)])
        self.ffn_norm = LayerNorm(hidden_channels, eps=1e-6)
        self.ffn = TranUnetMlp(hidden_channels, net_type=net_type)
        self.attn = RTBlock(hidden_channels, patch=patch)

    def forward(self, x, x_dual):
        raw = x
        b, p2, c= x.shape
        p = int(math.sqrt(p2))
        x = self.attention_norm[0](x)
        x_dual = self.attention_norm[1](x_dual)
        
        x = einops.rearrange(x, 'B (H W) C-> B C H W', H=p) 
        x_dual = einops.rearrange(x_dual, 'B (H W) C-> B C H W', H=p) 
        x, x_dual = self.attn(x, x_dual)
        x = einops.rearrange(x, 'B C H W-> B (H W) C') 
        x_dual = einops.rearrange(x_dual, 'B C H W-> B (H W) C')
        x = x + raw

        raw = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + raw
        return x, x_dual
    
class DualCrossTransformer(nn.Module):
    def __init__(self, in_channels, num_layer=1, patch=None, net_type='linear'):
        super().__init__()
        self.blocks=nn.ModuleList([
                DualCrossAttentionBlock(in_channels, patch=patch) for _ in range(num_layer)
            ])
        
    def forward(self, x, x_dual):
        b, c, h, w= x.shape
        x = einops.rearrange(x, 'B C H W-> B (H W) C') 
        x_dual = einops.rearrange(x_dual, 'B C H W-> B (H W) C')
        
        for i,b in enumerate(self.blocks):
            x, x_dual = b(x, x_dual)
            
        x = einops.rearrange(x, 'B (H W) C-> B C H W', H=h) 
        x_dual = einops.rearrange(x_dual, 'B (H W) C-> B C H W', H=h)

        return x, x_dual
        

class MultiScaleFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleFusion, self).__init__()

        self.myffs = nn.ModuleList([nn.Sequential(
                        DFCAM(inc, channel2wh(out_channels), channel2wh(out_channels), freq_sel_method = freq_method),
                        FSpatialAttention(transform='dct')
                      ) for inc in in_channels])

        self.upconv = nn.ModuleList([nn.Conv2d(inc, out_channels, 1) for inc in in_channels])
        
        self.conv = nn.Sequential(nn.Conv2d(len(in_channels)*out_channels, out_channels, kernel_size=1, bias=False),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace = True))


    def forward(self, x):
        # x = self.m_apply(raw, self.myffs)
        max_size = x[-1].shape[2]

        for i,_x in enumerate(x):
            x[i] = self.upsample(_x, max_size)
        x=self.m_apply(x, self.upconv)
        x = self.m_apply(x, self.myffs)
        x = self.conv(torch.cat((x), dim=1))
        
        return x
    
    def m_apply(self, x, module):
        return [module[i](j) for i, j in enumerate(x)]

    def upsample(self, x, size):
        x = F.interpolate(x, scale_factor=size//x.shape[2], mode="nearest")
        return x

class MultiScaleFreqAttention(nn.Module):
    def __init__(self, in_channels, out_channels, freq_method='dct8'):
        super(MultiScaleFreqAttention, self).__init__()

        self.myffs = nn.ModuleList([nn.Sequential(
                        DFCAM(out_channels, channel2wh(out_channels), channel2wh(out_channels), freq_sel_method = freq_method),
                        FSpatialAttention(transform='dct')
                      ) for inc in in_channels])
        
        self.upconv = nn.ModuleList([nn.Conv2d(inc, out_channels, 1) for inc in in_channels])
        
        self.conv = nn.Conv2d(len(in_channels)*out_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, features):
        max_size = features[-1].shape[2]
        
        for i,f in enumerate(features):
            features[i] = self.upsample(f, max_size)
        
        features=self.m_apply(features, self.upconv)
        features=self.m_apply(features, self.myffs)
        
        out = self.conv(torch.cat((features), dim=1))
        
        return out
    
    def m_apply(self, x, module):
        return [module[i](j) for i, j in enumerate(x)]

    def upsample(self, x, size):
        x = F.interpolate(x, scale_factor=size//x.shape[2], mode="nearest")
        return x

# class MultiScaleRelationFusion(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(MultiScaleFusion, self).__init__()
#
#         self.dct = nn.ModuleList([DualCrossTransformer(inc) for inc in in_channels])
#
#         self.upconv = nn.ModuleList([nn.Conv2d(inc, out_channels, 1) for inc in in_channels])
#
#         self.conv = nn.Sequential(nn.Conv2d(len(in_channels)*out_channels, out_channels, kernel_size=1, bias=False),
#                                   nn.BatchNorm2d(out_channels),
#                                   nn.ReLU(inplace = True))
#
#
#     def forward(self, x):
#         # x = self.m_apply(raw, self.myffs)
#         max_size = x[-1].shape[2]
#
#         for i,_x in enumerate(x):
#             x[i] = self.upsample(_x, max_size)
#         x=self.m_apply(x, self.upconv)
#         x = self.m_apply(x, self.myffs)
#         x = self.conv(torch.cat((x), dim=1))
#
#         return x
#
#     def m_apply(self, x, module):
#         return [module[i](j) for i, j in enumerate(x)]
#
#     def upsample(self, x, size):
#         x = F.interpolate(x, scale_factor=size//x.shape[2], mode="nearest")
#         return x


print('modules')