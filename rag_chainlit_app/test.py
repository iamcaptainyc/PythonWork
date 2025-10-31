import torch
import torch.nn as nn


class BatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(CustomBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # 初始化可学习的scale和shift参数  
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

        # 初始化运行均值和方差  
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)

        # 如果在训练模式下，使用这些变量来跟踪当前小批量的统计量  
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        if self.training:
            # 计算当前小批量的均值和方差  
            batch_mean = x.mean(dim=[0, 2, 3], keepdim=True)
            batch_var = x.var(dim=[0, 2, 3], keepdim=True)

            # 更新运行均值和方差  
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean.detach()
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var.detach()

            # 归一化数据  
            x_norm = (x - batch_mean) / torch.sqrt(batch_var + self.eps)

        else:
            # 使用运行均值和方差进行归一化  
            x_norm = (x - self.running_mean.unsqueeze(0).unsqueeze(2).unsqueeze(3)) / torch.sqrt(
                self.running_var.unsqueeze(0).unsqueeze(2).unsqueeze(3) + self.eps)

            # 应用可学习的scale和shift参数
        x_transformed = (self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) * x_norm +
                         self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3))

        # 更新跟踪的小批量数量  
        if self.training:
            self.num_batches_tracked += 1

        return x_transformed