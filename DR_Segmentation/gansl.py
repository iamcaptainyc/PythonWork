import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *

import wandb
from torch.autograd import Variable
import os
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt

import logger
from torch.utils.data import dataloader
from itertools import cycle

# from utils import utils
# from utils import metrics
# from loss import *

class Discriminator(nn.Module):
    def __init__(self, in_channels):
        channels = [32, 64, 128, 256]
        channels = [in_channels] + channels
        self.convs = nn.ModuleList([
                self.basic_conv(channels[i], channels[i+1])
                for i in range(len(channels)-1)
            ])
        self.fc = nn.Linear(channels[-1], 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
    def basic_conv(self, in_channels, out_channels):
        return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, 1),
                    nn.Conv2d(out_channels, out_channels, 3, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                )
        
    def forward(self, x):
        for m in self.convs:
            x = m(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return F.softmax(x)

class GANTrainer(SLTrainer):
    def __init__(self, seg_model, grading_model, optimizers, criterions, device, config):
        print("GAN trainer")
        self.ssl_name='GAN'
        self.config=config
        self.seg_model=seg_model  
        self.grading_model=grading_model
        self.optimizerS=optimizers[0]
        self.optimizerG=optimizers[1]
        self.device=device
        self.criterionS=criterions[0]
        self.criterionG=criterions[1]
        self.estimatorS=[Estimator(config.metrics, 2, binary=True, input_logits=config.input_logits) for _ in range(config.num_classes)]
        self.estimatorG=Estimator(config.gan_grading_metrics, config.gan_grading_classes, input_logits=config.input_logits)
        self.id=config.id
        
        
        self.discriminator=Discriminator(config.num_classes).to(self.device)
        self.optimizerD=torch.optim.SGD(self.discriminator.parameters(),lr=1e-3, momentum=0.9, weight_decay=0.0005)
        self.criterionD=torch.nn.BCELoss()
        
        self.data_dir=os.path.join(config.checkpoint_dir,'GAN trainer', self.id)
        self.log_dir=os.path.join(self.data_dir,'log')
        self.cpt_dir=os.path.join(self.data_dir,'checkpoint')
        self.best_checkpoint=''
        self.epoch       = 0
        self.start_epoch = 0
        self.last_epoch = -1
        mkdir(self.data_dir)
        mkdir(self.log_dir)
        mkdir(self.cpt_dir)
        
    def trn_iteration(self, seg_loader, grading_loader, scheduler):
        self.seg_model.train()
        self.grading_model.train()
        self.discriminator.train()
        
        for e in self.estimatorS:
            e.reset()
        self.estimatorG.reset()
        
        trn_seg_loss = AverageMeter()
        trn_grading_loss = AverageMeter()
        trn_dis_loss = AverageMeter()
        
        seg_loader=iter(cycle(seg_loader))
        
        lr=self.optimizer.state_dict()['param_groups'][0]['lr']
        with tqdm(total=len(grading_loader), desc=f"Epoch {self.epoch}/{self.config.num_epochs} --- training lr:{lr}", unit='batch') as pbar:
            for step, (x, y) in enumerate(grading_loader):
                Tensor = torch.cuda.FloatTensor
                valid = Variable(Tensor(x.size(0), 1).fill_(1.0), requires_grad=False)
                fake = Variable(Tensor(x.size(0), 1).fill_(0.0), requires_grad=False)

                x, y = [t.to(self.device) for t in (x,y)]
                x, y=Variable(x),Variable(y)
                
                s_x, s_y = next(seg_loader)
                s_x, s_y = [t.to(self.device) for t in (s_x,s_y)]
                s_x, s_y=Variable(s_x),Variable(s_y)
                
                # train seg and grading model
                pseudo_masks = self.seg_model(x.float())
                grading_pred, refined_pseudo_masks = self.grading_model(x.float(), pseudo_masks)
                s_y_pred = self.seg_model(s_x.float())
                
                s_y = torch.cat(s_y, refined_pseudo_masks)
                s_y_pred = torch.cat(s_y_pred, pseudo_masks)
                
                
                grading_loss = self.criterionG(grading_pred, y)
                trn_grading_loss.update(grading_loss.item())
                self.optimizerG.zero_grad()
                grading_loss.backward()
                self.optimizerG.step()
                self.estimatorG.update(grading_pred, y)
                
                # update seg estimator
                targets=[]
                for i in range(1,self.config.num_classes+1):
                    t=(s_y==i).float()
                    targets.append(t)
                    o=s_y_pred[:,i-1]
                    # print(f't:{t.shape}')
                    # print(f'o:{o.shape}')
                    self.estimatorS[i-1].update(o, t)
                targets=torch.stack(targets,dim=1)
                seg_loss = self.criterionS(s_y_pred, targets) + self.criterionD(self.discriminator(s_y_pred), valid)
                trn_seg_loss.update(seg_loss.item())
                
                self.optimizerS.zero_grad()
                seg_loss.backward()
                self.optimizerS.step()
                
                #train discriminator
                self.optimizerD.zero_grad()
                real_loss = self.criterionD(self.discriminator(s_y), valid)
                fake_loss = self.criterionD(self.discriminator(s_y_pred), fake)
                dis_loss = (real_loss + fake_loss) / 2
                trn_dis_loss.update(dis_loss)
                dis_loss.backward()
                self.optimizerD.step()
                
                
                # if self.config.accumulation_step:
                #     loss /= self.config.accumulation_step
                #     loss.backward()
                #
                #     if ((step + 1) % self.config.accumulation_step == 0) or (step + 1 == len(data_loader)):
                #         trn_loss.update(loss.item()*self.config.accumulation_step)
                #         self.optimizer.step()
                #         self.optimizer.zero_grad()
                # else: 
                #     trn_loss.update(loss.item())
                #     self.optimizer.zero_grad()
                #     loss.backward()
                #     self.optimizer.step()
                
                if not self.config.upd_by_ep and scheduler != None: scheduler.step()
                pbar.update(1)
        
        info_dict=dict([('epoch', self.epoch),
                   ('lr', lr),
                   ('trn_seg_loss', round(trn_seg_loss.avg,4)),
                   ('trn_grading_loss', round(trn_grading_loss.avg,4)),
                   ('trn_dis_loss', round(trn_dis_loss.avg,4))])

        for i,k in enumerate(self.config.leisions):
            score=self.estimatorS[i].get_scores(4)
            info_dict.update({'trn_seg_'+_k+'_'+k:_v for _k,_v in score.items()})
        
        grading_score = self.estimatorG.get_scores(4)
        info_dict.update({'trn_grading_'+k:v for k,v in grading_score.items()})
        
        return info_dict
        
    def val_iteration(self, seg_loader, grading_loader, val_loss):
        self.seg_model.eval()
        self.grading_model.eval()
        
        for e in self.estimatorS:
            e.reset()
        self.estimatorG.reset()
        
        with tqdm(total=len(seg_loader), desc=f"Epoch {self.epoch}/{self.config.num_epochs} --- validating", unit='batch') as pbar:
            with torch.no_grad():
                for step, (x, y) in enumerate(seg_loader):
                    x, y = [t.to(self.device) for t in (x,y)]
                    x, y=Variable(x),Variable(y)
                    y_pred = self.seg_model(x.float())
                        
                    #--------normal----------
                    targets=[]
                    for i in range(1,self.config.num_classes+1):
                        t=(y==i).float()
                        targets.append(t)
                        o=y_pred[:,i-1]
                        self.estimator[i-1].update(o, t)
                    targets=torch.stack(targets,dim=1)
                    loss = self.criterionS(y_pred, targets)
                    
                    # if self.config.accumulation_step:
                    #     loss /= self.config.accumulation_step
                    #     if ((step + 1) % self.config.accumulation_step == 0) or (step + 1 == len(data_loader)):
                    #         val_loss.update(loss.item()*self.config.accumulation_step)
                    # else: 
                    val_loss.update(loss.item())
                    pbar.update(1) 
                
        info_dict=dict([('epoch', self.epoch),
                   ('val_seg_loss', round(val_loss.avg,4))])

        mean_metrics={}
        for n in set(self.estimatorS[0].need_named_metrics)&set(self.estimatorS[0].get_scores(4).keys()):
            mean_metrics[n]=[]
        
        for i,k in enumerate(self.config.leisions):
            score=self.estimatorS[i].get_scores(4)
    
            for n in set(self.estimatorS[i].need_named_metrics)&set(score.keys()):
                mean_metrics[n].append(score[n])
        
            info_dict.update({'val_seg_'+_k+'_'+k:_v for _k,_v in score.items()})
            
        for k,v in mean_metrics.items():
            info_dict.update({'val_seg_'+k:np.round(np.mean(v), 4)})
            
        val_loss.reset()
        with tqdm(total=len(grading_loader), desc=f"Epoch {self.epoch}/{self.config.num_epochs} --- validating", unit='batch') as pbar:
            with torch.no_grad():
                for step, (x, y) in enumerate(grading_loader):
                    x, y = x.to(self.device), y.to(self.device)
                    x, y=Variable(x),Variable(y)
    
                    y_pred = self.grading_model(x.float())
    
                    loss=self.criterionG(y_pred, y)
                    val_loss.update(loss.item())
    
                    self.estimatorG.update(y_pred, y)
                    pbar.update(1)
                
        score=self.estimatorG.get_scores(4)
        
        info_dict.update(dict([('val_grading_loss', round(val_loss.avg,4))]))
        info_dict.update({'val_grading_'+k:v for k,v in score.items()})
            
        return info_dict

print('gansl.py')