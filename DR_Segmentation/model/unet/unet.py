from torch import nn
from torch.nn import functional as F
import torch
import einops
import math

class DualCrossAttentionBlock(nn.Module):
    def __init__(self, hidden_channels, net_type='linear'):
        super().__init__()
        self.hidden_size = hidden_channels
        self.attention_norm = nn.ModuleList([LayerNorm(hidden_channels, eps=1e-6) for _ in range(2)])
        self.ffn_norm = LayerNorm(hidden_channels, eps=1e-6)
        self.ffn = TranUnetMlp(hidden_channels, net_type=net_type)
        self.attn = RTBlock(hidden_channels)

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
        
        self.patch=patch
        if self.patch:
            self.patch_avg = nn.ModuleList([nn.AdaptiveAvgPool2d(output_size=(patch, patch)) for _ in range(2)]) #b h w c->b p*p c
            self.avg_map = nn.ModuleList([
                    nn.Sequential(nn.Conv2d(in_channels, in_channels, 1),
                                  nn.BatchNorm2d(in_channels), 
                                  nn.ReLU())
                    for _ in range(2)
                ])
        self.blocks=nn.ModuleList([
                DualCrossAttentionBlock(in_channels) for _ in range(num_layer)
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

class ConvRelu(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, is_up=True, scale_factor=2, mode='conv', fusion='cat'):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels
        self.is_up=is_up
        self.mode = mode
        self.fusion = fusion

        if mode == 'conv':
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2*scale_factor, stride=scale_factor, padding=scale_factor//2)
        elif mode == 'interp':
            self.up = nn.Upsample(scale_factor=scale_factor)
        
        self.conv = ConvRelu(in_channels + skip_channels, out_channels)
        
    def forward(self, x, skip=None):
        if skip is not None:
            if skip.shape[2] == x.shape[2]:
                x = x
            else:
                x = self.up(x)
        else:
            x = self.up(x)
            
            
        if skip != None:
            # print(f'x:{x.shape}')
            # print(f'skip:{skip.shape}')

            x = torch.cat([x, skip], axis=1)
            # print(f'cat_x:{x.shape}')
            
        x = self.conv(x)
            
        return x
    
class UNetResNet(nn.Module):
    def __init__(self, num_classes=1, pretrained=False, module_list = None, downsample_factor=8, upsample_mode='conv', dual=False, **kwargs):
        super(UNetResNet, self).__init__()

        self.args = kwargs['args']
        self.downsample_factor=downsample_factor
        self.module_list = module_list
        self.backbone=kwargs['backbone']
        self.resolution=kwargs['resolution']
        self.num_classes=num_classes
        self.dual=dual
        if self.backbone[:2] == 'rs':
            self.stages, encoder_channels=get_resnet_stages(self.downsample_factor, self.backbone, self.module_list, self.args)
        elif 'wtconvnext' in self.backbone:
            self.stages, encoder_channels=get_wtconvnext_stages(self.downsample_factor, self.backbone, self.args)
        self.stages=nn.ModuleList(self.stages)
        
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
            if self.args.seg_vessel:
                self.blocks_dual = nn.ModuleList([
                    DecoderBlock(in_ch, out_ch, sk_ch, mode=upsample_mode) for in_ch, out_ch, sk_ch in zip(self.in_channels, self.out_channels, self.skip_channels)
                ])
                self.final_dual = nn.Conv2d(decoder_channels[-1], 1, 1)
            else:
                self.stages_dual, _=get_resnet_stages(self.downsample_factor, self.backbone, self.module_list, self.args)
                self.stages_dual=nn.ModuleList(self.stages)
                if self.module_list:
                    if self.module_list.get('cross_att', False):
                        if self.module_list['cross_att'] == 'dct':
                            dual_cross_att=DualCrossTransformer
                        elif self.module_list['cross_att'] == 'rtb':
                            dual_cross_att=RTBlock
                    else:
                        dual_cross_att=RTBlock
                else:
                    dual_cross_att=RTBlock
                self.dual_cross_att=nn.ModuleList([dual_cross_att(in_c) for in_c in encoder_channels[1:3][::-1]])
            

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
                    self.msf=MultiScaleFusion(in_channels=encoder_channels[::-1])
                if self.dual and self.args.seg_vessel:
                    if self.module_list['msfm'] == 'rtb':
                        self.rtb=RTBlock(encoder_channels[0])
            if self.module_list.get('dam', False):
                if self.module_list['dam'] == 'msfa':
                    self.msfa=MultiScaleFreqAttention(decoder_channels[:-1], decoder_channels[-2], freq_method=self.module_list.get('freq_method','dct8'))
                if self.dual and self.args.seg_vessel:
                    if self.module_list['dam'] == 'rtb':
                        self.rtb=RTBlock(decoder_channels[-1], 16)
                if self.module_list['dam'] == 'msf':
                    self.msf=MultiScaleFusion(decoder_channels[:-1], decoder_channels[-2])
        
    
    def forward(self, x, x_dual=None):
        encoder_features=[]
        if self.dual and not self.args.seg_vessel:
            x_dual = x_dual.unsqueeze(1).repeat(1,3,1,1)
            for i,(m, m_dual) in enumerate(zip(self.stages, self.stages_dual)):
                x=m(x)
                x_dual=m_dual(x_dual)
                encoder_features.append(x)
                if i>=2 and i<4:
                    x, x_dual = self.dual_cross_att[i-2](x, x_dual)
        else:
            for m in self.stages:
                x=m(x)
                encoder_features.append(x)

        if self.module_list:
            encoder_features[-1] = self.att_modules(encoder_features[-1])
            if self.module_list.get('msfm', False):
                if self.module_list['msfm'] == 'dca':
                    encoder_features[:-1] = self.dca(encoder_features[:-1])
                elif self.module_list['msfm'] == 'msf':
                    encoder_features = self.msf(encoder_features)
                if self.dual and self.args.seg_vessel:
                    if self.module_list['msfm'] == 'rtb':
                        x, x_dual = self.rtb(x)
                        
        encoder_features = encoder_features[::-1]
        decoder_features=[]
        encoder_features=encoder_features[1:]
        head_feature = x

        for i, decoder_block in enumerate(self.blocks):
            if self.skip_channels[i] != 0:
                x = decoder_block(x, skip=encoder_features[i])
                decoder_features.append(x)
                
        if self.dual and self.args.seg_vessel:
            decoder_features_dual=[]
            if x_dual == None:
                x_dual = head_feature
            for i, decoder_block in enumerate(self.blocks_dual):
                if self.skip_channels[i] != 0:
                    x_dual = decoder_block(x_dual, skip=encoder_features[i])
                    decoder_features_dual.append(x_dual)
            
        if self.module_list:
            if self.module_list.get('dam', False):
                if self.module_list['dam'] == 'msfa':
                    x = self.msfa(decoder_features)
                if self.dual and self.args.seg_vessel:
                    if self.module_list['dam'] == 'rtb':
                        x, x_dual = self.rtb(x,x_dual)
                        
        x = self.blocks[-1](x)
        
        if self.dual and self.args.seg_vessel:
            x_dual = self.blocks_dual[-1](x_dual)
            return self.final(x), self.final_dual(x_dual)

        return self.final(x)


print('unet.py')