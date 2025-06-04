import torch
import torch.nn as nn
from torch.optim import *

import wandb
from torch.autograd import Variable
import os
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import logger
from torch.utils.data import dataloader


# from utils import utils
# from utils import metrics
# from loss import *

class MLSLTrainer(SLTrainer):
    def __init__(self, model, optimizer, criterion, device, config):
        print("Dual Branch Supervised trainer")
        self.ssl_name='MTSL'
        self.config=config
        self.model=model
        self.optimizer=optimizer
        self.device=device
        self.criterion=criterion
        self.estimator=[Estimator(config.metrics, 2, binary=True, input_logits=config.input_logits) for _ in range(config.num_classes)]
        
        self.id=config.id
        
        self.data_dir=os.path.join(config.checkpoint_dir,'Multi-Task Supervised trainer', self.id)
        self.log_dir=os.path.join(self.data_dir,'log')
        self.cpt_dir=os.path.join(self.data_dir,'checkpoint')
        self.best_checkpoint=''
        self.epoch       = 0
        self.start_epoch = 0
        self.last_epoch = -1
        mkdir(self.data_dir)
        mkdir(self.log_dir)
        mkdir(self.cpt_dir)
        
    def trn_iteration(self, data_loader, scheduler):
        self.model.train()
        for e in self.estimator:
            e.reset()
        trn_loss = AverageMeter()
        
        lr=self.optimizer.state_dict()['param_groups'][0]['lr']
        with tqdm(total=len(data_loader), desc=f"Epoch {self.epoch}/{self.config.num_epochs} --- training lr:{lr}", unit='batch') as pbar:
            for step, (x1, y) in enumerate(data_loader):
                x1, y = [t.to(self.device) for t in (x1,y)]
                x1, y=Variable(x1),Variable(y)
                
                y_pred = self.model(x1.float())

                #------separate_MA----------

                # ma_feature=y_pred[0][:,0]
                # y_pred=y_pred[0]
                
                # targets=[]
                # for i in range(1,self.config.num_classes+1):
                #     t=(y==i).float()
                #     targets.append(t)
                # targets=torch.stack(targets,dim=1)
                # ma_target=targets[:,1]
                # targets=targets[:,[True,False,True,True]]

                # self.estimator[0].update(ma_feature, ma_target)
                # for i in range(3):
                #     t=targets[:,i]
                #     o=y_pred[:,i]
                #     self.estimator[i+1].update(o, t)
                # loss = self.criterion(ma_feature, ma_target)
                # loss += self.criterion(y_pred, targets)

                #--------lseg------------
                if self.config.netframe=='lseg':
                    features=y_pred[:-1]
                    y_pred=y_pred[-1]
                    targets=[]
                    for i in range(1,self.config.num_classes+1):
                        t=(y==i).float()
                        targets.append(t)
                        o=y_pred[:,i-1]
                        self.estimator[i-1].update(o, t)
                    targets=torch.stack(targets,dim=1)
                    loss = self.criterion(y_pred, targets)
                    for f in features:
                        loss += self.criterion(f, targets)
                elif self.config.separate_form:
                    #----separate---------
                    targets=[]
                    for i in self.config.separate_form:
                        for j in i:
                            t=(y==j).float()
                            targets.append(t)
                    targets=torch.stack(targets,dim=1)
                    for i,e in enumerate(self.estimator):
                        e.update(y_pred[:,i],targets[:,i])
                    loss = self.criterion(y_pred, targets)
                else:
                    #--------normal----------
                    targets=[]
                    for i in range(1,self.config.num_classes+1):
                        t=(y==i).float()
                        targets.append(t)
                        o=y_pred[:,i-1]
                        # print(f't:{t.shape}')
                        # print(f'o:{o.shape}')
                        self.estimator[i-1].update(o, t)
                    targets=torch.stack(targets,dim=1)
                    loss = self.criterion(y_pred, targets)

                if self.config.accumulation_step:
                    loss /= self.config.accumulation_step
                    loss.backward()
                    
                    if ((step + 1) % self.config.accumulation_step == 0) or (step + 1 == len(data_loader)):
                        trn_loss.update(loss.item()*self.config.accumulation_step)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                else: 
                    trn_loss.update(loss.item())
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                if not self.config.upd_by_ep and scheduler != None: scheduler.step()
                pbar.update(1)
        
        info_dict=dict([('epoch', self.epoch),
                   ('lr', lr),
                   ('trn_loss', round(trn_loss.avg,4))])

        if self.config.separate_form:
            leisions = []
            for i in self.config.separate_form:
                for j in i:
                    leisions.append(self.config.leisions[j-1])
        else:
            leisions = self.config.leisions
        
        for i,k in enumerate(leisions):
            score=self.estimator[i].get_scores(4)
            if self.config.show_log:
                print(f'trn_gt_{k}:{self.estimator[i].count("gt")}')
                print(f'trn_pred_{k}:{self.estimator[i].count("pred")}')
        
            info_dict.update({'trn_'+_k+'_'+k:_v for _k,_v in score.items()})
        
        
        return info_dict
        
    def val_iteration(self, data_loader, val_loss):
        self.model.eval()
        
        for e in self.estimator:
            e.reset()
        
        with tqdm(total=len(data_loader), desc=f"Epoch {self.epoch}/{self.config.num_epochs} --- validating", unit='batch') as pbar:
            with torch.no_grad():
                for step, (x, y) in enumerate(data_loader):
                    x, y = x.to(self.device), y.to(self.device)
                    x, y=Variable(x),Variable(y)
    
                    y_pred = self.model(x.float())

                    #------separate_MA----------

                    # ma_feature=y_pred[0][:,0]
                    # y_pred=y_pred[0]
                    
                    # targets=[]
                    # for i in range(1,self.config.num_classes+1):
                    #     t=(y==i).float()
                    #     targets.append(t)
                    # targets=torch.stack(targets,dim=1)
                    # ma_target=targets[:,1]
                    # targets=targets[:,[True,False,True,True]]
    
                    # self.estimator[0].update(ma_feature, ma_target)
                    # for i in range(3):
                    #     t=targets[:,i]
                    #     o=y_pred[:,i]
                    #     self.estimator[i+1].update(o, t)
                    # loss = self.criterion(ma_feature, ma_target)
                    # loss += self.criterion(y_pred, targets)

                    #--------lseg------------
                    if self.config.netframe=='lseg':
                        features=y_pred[:-1]
                        y_pred=y_pred[-1]
                        targets=[]
                        for i in range(1,self.config.num_classes+1):
                            t=(y==i).float()
                            targets.append(t)
                            o=y_pred[:,i-1]
                            self.estimator[i-1].update(o, t)
                        targets=torch.stack(targets,dim=1)
                        loss = self.criterion(y_pred, targets)
                        for f in features:
                            loss += self.criterion(f, targets)
                    elif self.config.separate_form:
                        #----separate---------
                        targets=[]
                        for i in self.config.separate_form:
                            for j in i:
                                t=(y==j).float()
                                targets.append(t)
                        targets=torch.stack(targets,dim=1)
                        for i,e in enumerate(self.estimator):
                            e.update(y_pred[:,i],targets[:,i])
                        loss = self.criterion(y_pred, targets)
                    else:
                        #--------normal----------
                        targets=[]
                        for i in range(1,self.config.num_classes+1):
                            t=(y==i).float()
                            targets.append(t)
                            o=y_pred[:,i-1]
                            self.estimator[i-1].update(o, t)
                        targets=torch.stack(targets,dim=1)
                        loss = self.criterion(y_pred, targets)

                    if self.config.accumulation_step:
                        loss /= self.config.accumulation_step
                        if ((step + 1) % self.config.accumulation_step == 0) or (step + 1 == len(data_loader)):
                            val_loss.update(loss.item()*self.config.accumulation_step)
                    else: 
                        val_loss.update(loss.item())
                    pbar.update(1) 
                
        info_dict=dict([('epoch', self.epoch),
                   ('val_loss', round(val_loss.avg,4))])

        mean_metrics={}
        for n in set(self.estimator[0].need_named_metrics)&set(self.estimator[0].get_scores(4).keys()):
            mean_metrics[n]=[]

        if self.config.separate_form:
            leisions = []
            for i in self.config.separate_form:
                for j in i:
                    leisions.append(self.config.leisions[j-1])
        else:
            leisions = self.config.leisions
        
        for i,k in enumerate(leisions):
            score=self.estimator[i].get_scores(4)
    
            for n in set(self.estimator[i].need_named_metrics)&set(score.keys()):
                mean_metrics[n].append(score[n])
                
            if self.config.show_log:
                print(f'val_gt_{k}:{self.estimator[i].count("gt")}')
                print(f'val_pred_{k}:{self.estimator[i].count("pred")}')
        
            info_dict.update({'val_'+_k+'_'+k:_v for _k,_v in score.items()})
            
        for k,v in mean_metrics.items():
            info_dict.update({'val_'+k:np.round(np.mean(v), 4)})
            
        return info_dict

print('mlsl.py')