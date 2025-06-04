from torch import nn
from torch.nn import functional as F
import torch


class ResNetLseg(nn.Module):
    def __init__(self, num_classes=1, pretrained=False, module_list = None, downsample_factor=8, task_num=4, **kwargs):
        super(ResNetLseg, self).__init__()
        
        self.task_num=task_num
        self.downsample_factor=downsample_factor
        self.module_list = module_list
        self.backbone=kwargs['backbone']
        self.rsd=[self.downsample_factor<(2**i) for i in range(3,6)]
        if self.backbone == 'rs50':
            self.encoder = resnet50(pretrained=pretrained, replace_stride_with_dilation=self.rsd)
        elif self.backbone == 'rs101':
            self.encoder = resnet101(pretrained=pretrained, replace_stride_with_dilation=self.rsd)
        elif self.backbone == 'rs152':
            self.encoder = resnet152(pretrained=pretrained, replace_stride_with_dilation=self.rsd)
        else:
            assert f'{self.backbone} does not exists!'
        
        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu)#x/2
        self.conv1_out_c=64
        self.supervision1=nn.Sequential(nn.Conv2d(self.conv1_out_c,self.task_num,1),
                                        nn.Upsample(scale_factor=2))
        
        self.conv2=nn.Sequential(self.encoder.maxpool,
                                 self.encoder.layer1)#x/4
        self.conv2_out_c=256
        self.supervision2=nn.Sequential(nn.Conv2d(self.conv2_out_c,self.task_num,1),
                                        nn.Upsample(scale_factor=4))
        
        self.conv3=nn.Sequential(self.encoder.layer2)#x/8
        self.conv3_out_c=512
        self.supervision3=nn.Sequential(nn.Conv2d(self.conv3_out_c,self.task_num,1),
                                        nn.Upsample(scale_factor=8))
        
        self.conv4=nn.Sequential(self.encoder.layer3)#x/16
        self.conv4_out_c=1024
        self.supervision4=nn.Sequential(nn.Conv2d(self.conv4_out_c,self.task_num,1),
                                        nn.Upsample(scale_factor=min(16,self.downsample_factor)))
        
        self.conv5=nn.Sequential(self.encoder.layer4)
        self.conv5_out_c=2048
        self.supervision5=nn.Sequential(nn.Conv2d(self.conv5_out_c,self.task_num,1),
                                        nn.Upsample(scale_factor=min(32,self.downsample_factor)))
        
        self.conv1x1=nn.ModuleList([nn.Conv2d(5,1,1) for _ in range(self.task_num)])
        
        # self.sigmoid=nn.Sigmoid()
        # self.final = nn.Conv2d(4, num_classes, kernel_size=1)

    
    def forward(self, x):
        conv1 = self.conv1(x)#x/2
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)#x/32
        
        feature1 = self.supervision1(conv1)
        feature2 = self.supervision2(conv2)
        feature3 = self.supervision3(conv3)
        feature4 = self.supervision4(conv4)
        feature5 = self.supervision5(conv5)
#         print(f'feature1:{feature1.shape}')
        
        cw = []
        for i in range(self.task_num):
            cw.append(torch.cat((feature1[:,i:i+1],feature2[:,i:i+1],feature3[:,i:i+1],feature4[:,i:i+1],feature5[:,i:i+1]),dim=1))
        
        for i,(c,m) in enumerate(zip(cw,self.conv1x1)):
            cw[i] = m(c)
            
        out = torch.cat((cw), dim=1)
        
        return feature1, feature2, feature3, feature4, feature5, out

print('lseg.py')