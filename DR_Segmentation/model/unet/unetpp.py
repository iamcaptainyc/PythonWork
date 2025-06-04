import torch
import torch.nn as nn
import torch.nn.functional as F
    
class UnetPlusDecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels
    ):
        super().__init__()
        self.conv = ConvRelu(in_channels+skip_channels, out_channels)

    def forward(self, x, skip=None):
        if skip is not None:
            # print(f'skip:{skip.shape}')
            # print(f'x:{x.shape}')
            if skip.shape[2] == x.shape[2]:
                x = x
            else:
                x = F.interpolate(x, scale_factor=2, mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class UnetPP(nn.Module):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__()
        self.args=kwargs['args']
        self.resolution=kwargs['resolution']
        self.downsample_factor= self.args.downsample_factor
        self.module_list = self.args.module_list
        self.backbone= self.args.backbone
        self.num_classes= self.args.num_classes
        self.dual=self.args.dual
        upsample_mode=self.args.upsample_mode
        
        if self.backbone[:2] == 'rs':
            self.stages, encoder_channels=get_resnet_stages(self.downsample_factor, self.backbone, self.module_list, self.args)
        elif 'wtconvnext' in self.backbone:
            self.stages, encoder_channels=get_wtconvnext_stages(self.downsample_factor, self.backbone, self.args)
        
        self.stages=nn.ModuleList(self.stages)
        # encoder_channels = [self.conv1_out_c, self.conv2_out_c, self.conv3_out_c, self.conv4_out_c, self.conv5_out_c]
        decoder_channels = [1024,512,256,128,64,32]
        decoder_channels = decoder_channels[6-len(encoder_channels):]
        
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        self.in_channels = [head_channels] + list(decoder_channels[:-1]) # 2048, 256, 128, 64, 32, 16
        self.skip_channels = list(encoder_channels[1:]) + [0] # 1024,512,128,64,0
        self.out_channels = decoder_channels 

        blocks = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(layer_idx + 1):
                if depth_idx == 0:
                    in_ch = self.in_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1)
                    out_ch = self.out_channels[layer_idx]
                else:
                    out_ch = self.skip_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (
                        layer_idx + 1 - depth_idx
                    )
                    in_ch = self.skip_channels[layer_idx - 1]
                blocks[f"x_{depth_idx}_{layer_idx}"] = DecoderBlock(
                    in_ch, out_ch, skip_ch, mode=self.args.upsample_mode
                )
        blocks[f"x_{0}_{len(self.in_channels)-1}"] = DecoderBlock(
            self.in_channels[-1], self.out_channels[-1], 0, mode=self.args.upsample_mode, scale_factor=2 if len(encoder_channels) == 5 else 4
        )

        self.final = nn.Conv2d(decoder_channels[-1], self.num_classes, kernel_size=1)
        
        self.blocks = nn.ModuleDict(blocks)
        self.depth = len(self.in_channels) - 1
        
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
                    
        features = encoder_features[::-1]
        encoder_features = encoder_features[::-1]
        # print('features:\n')
        # for i,f in enumerate(features):
        #     print('features[{}].shape={}'.format(i, f.shape))
            
        # start building dense connections
        dense_x = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(self.depth - layer_idx):
                if layer_idx == 0:
                    output = self.blocks[f"x_{depth_idx}_{depth_idx}"](
                        features[depth_idx], features[depth_idx + 1]
                    )
                    # print('\ndense_x[x_{}_{}].shape={},\n input:features[{}].shape={},\n skip:features[{}].shape={}'.format(
                    #         depth_idx, depth_idx, output.shape, 
                    #         depth_idx, features[depth_idx].shape,
                    #         depth_idx+1, features[depth_idx + 1].shape))
                    dense_x[f"x_{depth_idx}_{depth_idx}"] = output
                else:
                    dense_l_i = depth_idx + layer_idx
                    cat_names=['dense[x_{}_{}].shape={}'.format(idx, dense_l_i, dense_x[f"x_{idx}_{dense_l_i}"].shape) for idx in range(depth_idx + 1, dense_l_i + 1)]
                    # print(f'cat_features:\n--------\n{cat_names}\n--------\n')
                    cat_features = [
                        dense_x[f"x_{idx}_{dense_l_i}"]
                        for idx in range(depth_idx + 1, dense_l_i + 1)
                    ]
                    # print('cat_features+=features[{}].shape={}\n'.format(dense_l_i + 1, features[dense_l_i + 1].shape))
                    cat_features = torch.cat(
                        cat_features + [features[dense_l_i + 1]], dim=1
                    )
                    dense_x[f"x_{depth_idx}_{dense_l_i}"] = self.blocks[
                        f"x_{depth_idx}_{dense_l_i}"
                    ](dense_x[f"x_{depth_idx}_{dense_l_i-1}"], cat_features)
                    # print('dense_x[x_{}_{}].shape={},\n input:dense_x[f"x_{}_{}"].shape={},\n skip:cat_features.shape={}'.format(
                    #         depth_idx, dense_l_i, dense_x[f"x_{depth_idx}_{dense_l_i}"].shape, 
                    #         depth_idx, dense_l_i-1, dense_x[f"x_{depth_idx}_{dense_l_i-1}"].shape,
                    #         cat_features.shape))
        dense_x[f"x_{0}_{self.depth}"] = self.blocks[f"x_{0}_{self.depth}"](
            dense_x[f"x_{0}_{self.depth-1}"]
        )
        # print('dense_x[x_{}_{}].shape={}, input:dense_x[f"x_{}_{}"].shape={}'.format(
        #                     0, self.depth, dense_x[f"x_{0}_{self.depth}"].shape, 
        #                     0, self.depth-1, dense_x[f"x_{0}_{self.depth-1}"].shape))

        if self.dual and self.args.seg_vessel:
            decoder_features_dual=[]
            if x_dual == None:
                x_dual = encoder_features[0]
                encoder_features=encoder_features[1:]
            for i, decoder_block in enumerate(self.blocks_dual):
                if self.skip_channels[i] != 0:
                    # print('encoder_features[{}].shape={}'.format(i, encoder_features[i].shape))
                    x_dual = decoder_block(x_dual, skip=encoder_features[i])
                    decoder_features_dual.append(x_dual)

        decoder_features=[dense_x[f'x_{0}_{d}'] for d in range(self.depth)]
        # for df in decoder_features:
        #     print(df.shape)

        x = dense_x[f'x_{0}_{self.depth}']
        
        if self.module_list:
            if self.module_list.get('dam', False):
                if self.module_list['dam'] == 'msfa':
                    x = self.msfa(decoder_features)
                    x = self.blocks[f"x_{0}_{self.depth}"](x)
                if self.module_list['dam'] == 'msf':
                    x = self.msf(decoder_features)
                    x = self.blocks[f"x_{0}_{self.depth}"](x)
                    
        
        
        if self.dual and self.args.seg_vessel:
            x_dual = self.blocks_dual[-1](x_dual)
            return self.final(x), self.final_dual(x_dual)
        
        out = self.final(x)
        return out

print('unetpp.py')