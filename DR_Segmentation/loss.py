#!coding:utf-8
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch.nn.modules.loss import BCEWithLogitsLoss

# from const import *

def get_criterion(args, **kwargs):
    if args.loss_method == 'ce':
        criterion=torch.nn.CrossEntropyLoss()
    elif args.loss_method == 'wce':
        criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor(args.ce_weight).to(kwargs['device']))
    elif args.loss_method == 'dice':
        criterion=DiceLoss()
    elif args.loss_method == 'ce_dice':
        criterion=CE_DiceLoss(weight=torch.tensor(args.ce_weight).to(kwargs['device']))
    elif args.loss_method == 'focal':
        criterion=FocalLoss(gamma=args.focal_gamma, alpha=torch.tensor(args.ce_weight).to(kwargs['device']))
    elif args.loss_method == 'bce':
        weight=torch.tensor(args.bce_weight).to(kwargs['device'])
        criterion=nn.BCEWithLogitsLoss(pos_weight=weight)
    elif args.loss_method == 'bce_dice':
        weight=torch.tensor(args.bce_weight).to(kwargs['device'])
        criterion=BCE_DiceLoss(weight=weight)
    elif args.loss_method == 'wbce':
        weight=torch.tensor(args.bce_weight).to(kwargs['device'])
        criterion=WeightedBCE(weight=weight)
    elif args.loss_method == 'cwbce':
        criterion=CustomWeightedLoss()
    return criterion

#学习率调整
def get_scheduler(args, optimizer, last_epoch=-1):
    if args.scheduler == 'cos':
        scheduler= lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.num_epochs,eta_min=args.min_lr,last_epoch=last_epoch)
    elif args.scheduler == 'lam':
        def ad_lr(epoch):
            t=8
            if epoch <=t:
                return 1
            else:
                return math.pow(0.1,((epoch-t)//4)+1)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=ad_lr,last_epoch=last_epoch)
    elif args.scheduler == 'multi':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_gamma,last_epoch=last_epoch)
    elif args.scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.lr_gamma,last_epoch=last_epoch)
    else:
        scheduler=None
    return scheduler

def make_one_hot(labels, classes):
    one_hot = torch.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target

class CustomWeightedLoss(nn.Module):
    def __init__(self):
        super(CustomWeightedLoss, self).__init__()
        self.beta = 0.8

    def forward(self, preds, targets):
        """
        Args:
            preds (Tensor): 模型预测的概率，形状为 (batch_size, c, H, W)。
            targets (Tensor): 目标标签，形状为 (batch_size, c, H, W)。
        """
        preds = torch.clamp(preds,min=1e-8,max=1-1e-8)
        preds = F.sigmoid(preds)
        
        self.beta=(targets==0).sum(dim=(0,2,3))/(targets.shape[0]*targets.shape[2]*targets.shape[3])
        self.beta=self.beta.view(1,targets.shape[1],1,1)
        # 计算每个类别的损失
        bce = - self.beta * targets * torch.log(preds) - (1 - targets) * (1-self.beta) * torch.log(1 - preds)
        
        loss = torch.mean(bce)
        # print(self.beta)
        # print(f'(targets==0).sum(dim=(0,2,3)):{(targets==0).sum(dim=(0,2,3))}')
        # print(bce.shape)
        # print(loss)
        # print(loss.item())
        return loss

class WeightedBCE(nn.Module):
    def __init__(self, weight):
        super(WeightedBCE, self).__init__()
        self.weight=weight

    def forward(self, output, target):
        batch_weight = self.weight[target.data.view(-1).long()].view_as(target)
        bce = nn.BCELoss(weight=batch_weight)
        return bce(output, target)
        
        
class DiceLoss(nn.Module):
    def __init__(self, smooth=1., ignore_index=255, binary=False):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.binary=binary

    def forward(self, output, target):
        if self.binary:
            output = F.sigmoid(output)
            output = output.contiguous().view(-1)
            target = target.contiguous().view(-1)
            
            pos_inter=output*target
            neg_inter=(1-output)*(1-target)
            intersection = pos_inter.sum()+neg_inter.sum()
            loss = 1-((2 * intersection + self.smooth) / (2*target.numel()+ self.smooth))
        else:
            if self.ignore_index not in range(target.min(), target.max()):
                if (target == self.ignore_index).sum() > 0:
                    target[target == self.ignore_index] = target.min()
            target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
            output = F.softmax(output, dim=1)
            output_flat = output.contiguous().view(-1)
            target_flat = target.contiguous().view(-1)
            intersection = (output_flat * target_flat).sum()
            loss = 1 - ((2. * intersection + self.smooth) /
                        (output_flat.sum() + target_flat.sum() + self.smooth))
        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, ignore_index=255, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.CE_loss = nn.CrossEntropyLoss(reduce=False, ignore_index=ignore_index, weight=alpha)

    def forward(self, output, target):
        logpt = self.CE_loss(output, target)
        pt = torch.exp(-logpt)
        loss = ((1-pt)**self.gamma) * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()

class CE_DiceLoss(nn.Module):
    def __init__(self, reduction='mean', ignore_index=255, weight=None):
        super(CE_DiceLoss, self).__init__()
        self.dice = DiceLoss()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)
    
    def forward(self, output, target):
        CE_loss = self.cross_entropy(output, target)
        dice_loss = self.dice(output, target)
        return CE_loss + dice_loss

class BCE_DiceLoss(nn.Module):
    def __init__(self, weight, smooth=1):
        super(BCE_DiceLoss, self).__init__()
        self.dice = DiceLoss(smooth = smooth, binary=True)
        self.bce = BCEWithLogitsLoss(pos_weight=weight)
        
    def forward(self, output, target):
        bce_loss = self.bce(output, target)
        dice_loss = self.dice(output, target)
        return bce_loss+dice_loss
        

print('loss.py')