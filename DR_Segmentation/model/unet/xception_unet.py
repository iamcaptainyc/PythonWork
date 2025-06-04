import torch
import torch.nn as nn
import torch.nn.functional as F

import pywt
import pywt.data
import torch.nn.functional as F


def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters

def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x

class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1,in_channels,1,1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels*4, in_channels*4, kernel_size, padding='same', stride=1, dilation=1, groups=in_channels*4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1,in_channels*4,1,1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.do_stride = nn.AvgPool2d(kernel_size=1, stride=stride)
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = wavelet_transform(curr_x_ll, self.wt_filter)
            curr_x_ll = curr_x[:,:,0,:,:]
            
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:,:,0,:,:])
            x_h_in_levels.append(curr_x_tag[:,:,1:4,:,:])

        next_x_ll = 0

        for i in range(self.wt_levels-1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = inverse_wavelet_transform(curr_x, self.iwt_filter)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0
        
        x = self.base_scale(self.base_conv(x))
        x = x + x_tag
        
        if self.do_stride is not None:
            x = self.do_stride(x)

        return x

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None
    
    def forward(self, x):
        return torch.mul(self.weight, x)

class conv_block(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1),
                 dilation=(1, 1),
                 ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation)

        self.norm = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

class DWconv(nn.Module):
    def __init__(self, 
                in_features, 
                out_features,
                kernel_size=3,
                stride=1, 
                padding=1, 
                dilation=1,
                pointwise=False,
                dwconv='conv',
                wt_levels=3
                ):
        super().__init__()
        self.pointwise = pointwise
        if dwconv == 'conv':
            self.depthwise = nn.Conv2d(
                in_channels=in_features,
                out_channels=in_features if pointwise else out_features,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_features,
                dilation=dilation)
        elif dwconv == 'wtconv':
            self.depthwise = WTConv2d(
                in_channels=in_features,
                out_channels=in_features if pointwise else out_features,
                kernel_size=kernel_size,
                stride=stride,
                wt_levels=wt_levels
                )
        if pointwise:
            self.pointwise = nn.Conv2d(in_features, 
                                        out_features, 
                                        kernel_size=(1, 1), 
                                        stride=(1, 1), 
                                        padding=(0, 0),
                                        dilation=(1, 1))

        self.norm = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.depthwise(x)
        if self.pointwise:
            x = self.pointwise(x)
        x = self.norm(x)
        x = self.relu(x)
        return x
    
class SeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, dwconv='conv', wt_levels=3):
        super().__init__()
        
        self.branch1=nn.Sequential(
                DWconv(in_channels, out_channels, dwconv=dwconv, wt_levels=wt_levels, pointwise=True),
                DWconv(out_channels, out_channels, dwconv=dwconv, wt_levels=wt_levels),
                nn.MaxPool2d(2,2)
            )
        
        self.branch2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.MaxPool2d(2,2)          
            )
        
    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        
        return x1+x2
    
class DWUnet(nn.Module):
    def __init__(self, num_classes=1, pretrained=False, module_list = None, downsample_factor=8, upsample_mode='interp', dwconv='dwconv', wt_levels=[5,4,3,2], dual=None, **kwargs):
        super(DWUnet, self).__init__()
        
        self.dual=dual
        self.downsample_factor=downsample_factor
        self.module_list = module_list
        
        self.conv1 = nn.Sequential(
                ConvRelu(3, 32),
                nn.MaxPool2d(2,2) #x/2
            )
        self.conv1_out_c=32
        
        self.conv2=SeparableConv(32, 64, dwconv=dwconv, wt_levels=wt_levels[0])#x/4
        self.conv2_out_c=64
        
        self.conv3=SeparableConv(64, 128, dwconv=dwconv, wt_levels=wt_levels[1])#x/8
        self.conv3_out_c=128
        
        self.conv4=SeparableConv(128, 256, dwconv=dwconv, wt_levels=wt_levels[2])#x/16
        self.conv4_out_c=256

        self.conv5=SeparableConv(256, 512, dwconv=dwconv, wt_levels=wt_levels[3])#x/32
        self.conv5_out_c=512
        
        stages = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]
        self.stages=nn.ModuleList(stages)
        encoder_channels = [self.conv1_out_c, self.conv2_out_c, self.conv3_out_c, self.conv4_out_c, self.conv5_out_c]
        
        
        decoder_channels = [1024,512,256,128,64,32]
        decoder_channels = decoder_channels[6-len(encoder_channels):]
        
        encoder_channels = encoder_channels[::-1]
        
        head_channels = encoder_channels[0]
        self.in_channels = [head_channels] + list(decoder_channels[:-1]) # 2048, 256, 128, 64, 32, 16
        self.skip_channels = list(encoder_channels[1:]) + [0] # 1024,512,128,64,0
        self.out_channels = decoder_channels 
        
        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch, mode=upsample_mode) for in_ch, out_ch, sk_ch in zip(self.in_channels, self.out_channels, self.skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        
        if self.dual:
            self.blocks_dual = nn.ModuleList([
                    DecoderBlock(in_ch, out_ch, sk_ch, mode=upsample_mode) for in_ch, out_ch, sk_ch in zip(self.in_channels, self.out_channels, self.skip_channels)
                ])
            self.final_dual = nn.Conv2d(decoder_channels[-1], 1, 1)

        self.final = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)
     
        if self.module_list:
            self.att_modules = nn.Sequential(*get_modules(self.module_list, head_channels))
            if self.module_list.get('msfm', False):
                if self.module_list['msfm'] == 'dca':
                    patch=int(self.resolution/min(16,self.downsample_factor))#应该变为crop_size=512，resolution是1024
                    self.dca=DCA(n=1,                                            
                                 features = encoder_channels[::-1][:-1],                                                                                                              
                                 patch=patch)
                elif self.module_list['msfm'] == 'msf':
                    patch=int(self.resolution/self.downsample_factor)
                    self.msf=MultiScaleFusion(in_channels=encoder_channels[::-1], patch=patch)
                if self.dual:  
                    if self.module_list['msfm'] == 'rtb':
                        self.rtb=RTBlock(encoder_channels[0])
            if self.module_list.get('dam', False):
                if self.module_list['dam'] == 'msfa':
                    self.msfa=MultiScaleFreqAttention(decoder_channels[:-1], decoder_channels[-2], freq_method=self.module_list.get('freq_method','dct8'))

    
    
    def forward(self, x):
        encoder_features=[]
        
        for m in self.stages:
            x=m(x)
            encoder_features.append(x)

        if self.module_list:
            encoder_features[-1] = self.att_modules(encoder_features[-1])
            if self.module_list.get('msfm', False):
                if self.module_list['msfm'] == 'dca':
                    encoder_features[:-1] = self.dca(encoder_features[:-1])
                elif self.module_list['msfm'] == 'msf':
                    encoder_features[-1] = self.msf(encoder_features)
                if self.dual:
                    if self.module_list['msfm'] == 'rtb':
                        x, x_dual = self.rtb(x)
                    
        encoder_features = encoder_features[::-1]
        
        decoder_features=[]
        encoder_features=encoder_features[1:]

        for i, decoder_block in enumerate(self.blocks):
            if self.skip_channels[i] != 0:
                x = decoder_block(x, skip=encoder_features[i])
                decoder_features.append(x)
        x = self.blocks[-1](x)
                
        if self.dual:
            decoder_features_dual=[]
            for i, decoder_block in enumerate(self.blocks_dual):
                if self.skip_channels[i] != 0:
                    x_dual = decoder_block(x_dual, skip=encoder_features[i])
                    decoder_features_dual.append(x_dual)
            x_dual = self.blocks_dual[-1](x_dual)
            
        if self.module_list:
            if self.module_list.get('dam', False):
                if self.module_list['dam'] == 'msfa':
                    x = self.msfa(decoder_features)
        
        if self.dual:
            return self.final(x), self.final_dual(x_dual)

        return self.final(x)
    
print('DWUnet.py')