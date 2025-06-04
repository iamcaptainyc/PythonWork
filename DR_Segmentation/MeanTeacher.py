import torch
import torch.nn as nn
from torch.optim import *

from sklearn.metrics import *
import wandb
from torch.autograd import Variable
import os
from tqdm import tqdm
from sklearn.metrics._classification import accuracy_score
import copy

import logger

# from utils import *
# from loss import *
# from ramps import *
# from mixup import *
# from loss import *
# from metrics import *

class MTTrainer:
    def __init__(self, model, ema_model, optimizer, criterion, device, config):
        print("MeanTeacher trainer")
        self.ssl_name='MT'
        self.config=config
        self.model=model
        self.ema_model=ema_model
        self.optimizer=optimizer
        self.device=device
        self.criterion=criterion
        self.estimator=Estimator(config.metrics, config.num_classes, config.loss_method)
        self.u_estimator=Estimator(config.metrics, config.num_classes, config.loss_method)
        self.ema_estimator=Estimator(config.metrics, config.num_classes, config.loss_method)
        self.cons_loss=mse_with_softmax
        
        self.id=config.id
        
        self.data_dir=os.path.join(config.checkpoint_dir,'MeanTeacher')
        self.log_dir=os.path.join(self.data_dir,'log')
        self.cpt_dir=os.path.join(self.data_dir,'checkpoint')
        self.usp_weight  = config.usp_weight
        self.ema_decay   = config.ema_decay
        self.rampup      = exp_rampup(config.weight_rampup)
        self.global_step = 0
        self.epoch       = 0
        
    def trn_iteration(self, data_loader, scheduler):
        self.model.train()
        self.ema_model.train()
        
        print(f'class_weight:{data_loader.dataset.class_weight}')
        self.estimator.reset()
        self.u_estimator.reset()
        self.ema_estimator.reset()
        trn_l_loss = AverageMeter()
        trn_u_loss = AverageMeter()
        
        lr=self.optimizer.state_dict()['param_groups'][0]['lr']
        with tqdm(total=len(data_loader), desc=f"Epoch {self.epoch}/{self.config.num_epochs} --- training lr:{lr}", unit='batch') as pbar:
            for step, ((x1,x2), y) in enumerate(data_loader):
                x1, x2, y = [t.to(self.device) for t in (x1,x2,y)]
                x1, x2, y=Variable(x1),Variable(x2),Variable(y)
                self.global_step+=1
#                 lmask, umask = self.decode_targets(y)
                
                l_len=self.config.sub_batch_size
                y_pred = self.model(x1.float())
                loss = self.criterion(y_pred[:l_len], y[:l_len])
                trn_l_loss.update(loss.item())
                
                self.update_ema(self.model, self.ema_model, self.ema_decay, self.global_step)
                
                with torch.no_grad():
                    ema_y_pred = self.ema_model(x2.float()) #将第二种图像增强后的数据输入教师模型
                    ema_y_pred = ema_y_pred.detach()
                
                cons_loss  = self.cons_loss(y_pred, ema_y_pred) #计算第一种图像增强与第二种图像增强预测之间的损失
                cons_loss *= self.rampup(self.epoch)*self.usp_weight
                loss += cons_loss
                trn_u_loss.update(cons_loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if not self.config.upd_by_ep and scheduler != None: scheduler.step()
                self.estimator.update(y_pred[:l_len], y[:l_len])
                self.u_estimator.update(y_pred[l_len:], y[l_len:])
                self.ema_estimator.update(ema_y_pred, y)
                
                pbar.update(1)
                
        score=self.estimator.get_scores(4)
        u_score=self.u_estimator.get_scores(4)
        ema_score=self.ema_estimator.get_scores(4)
        
        print(f'trn_l_gt:{self.estimator.count("gt")}')
        print(f'trn_l_pred:{self.estimator.count("pred")}')
        print(f'trn_u_gt:{self.u_estimator.count("gt")}')
        print(f'trn_u_pred:{self.u_estimator.count("pred")}')
        print(f'trn_uema_pred:{self.ema_estimator.count("pred")}')
        
        
        info_dict=dict([('epoch', self.epoch),
                   ('lr', lr),
                   ('trn_l_loss', round(trn_l_loss.avg,4)),
                   ('trn_u_loss', round(trn_u_loss.avg,4))])
        
        info_dict.update({'trn_l_'+k:v for k,v in score.items()})
        info_dict.update({'trn_u_'+k:v for k,v in u_score.items()})
        info_dict.update({'trn_ema_'+k:v for k,v in ema_score.items()})
        
        return info_dict
        
    def val_iteration(self, data_loader, val_loss):
        self.model.eval()
        self.ema_model.eval()
        
        self.estimator.reset()
        self.ema_estimator.reset()
        with tqdm(total=len(data_loader), desc=f"Epoch {self.epoch}/{self.config.num_epochs} --- validating", unit='batch') as pbar:
            with torch.no_grad():
                for step, (x, y) in enumerate(data_loader):
                    x, y = x.to(self.device), y.to(self.device)
                    x, y=Variable(x),Variable(y)
    
                    y_pred = self.model(x.float())
                    ema_y_pred = self.ema_model(x.float())
    
                    loss=self.criterion(y_pred, y)
                    val_loss.update(loss.item())
    
                    self.estimator.update(y_pred, y)
                    self.ema_estimator.update(ema_y_pred, y)
                    
                    pbar.update(1) 
        
        score=self.estimator.get_scores(4)
        ema_score=self.ema_estimator.get_scores(4)
        
        print(f'val_gt:{self.estimator.count("gt")}')
        print(f'val_pred:{self.estimator.count("pred")}')
        print(f'val_uema_pred:{self.ema_estimator.count("pred")}')
        
        info_dict=dict([('val_loss', round(val_loss.avg,4))])
        info_dict.update({'val_'+k:v for k,v in score.items()})
        info_dict.update({'val_ema_'+k:v for k,v in ema_score.items()})
        
        return info_dict
    
    def loop(self, num_epochs, trn_loader, val_loader, scheduler=None):
        logger.configure(dir=self.log_dir, log_suffix=self.id)
        
        if self.config.use_wandb:
            wandb.init(
              project=self.config.experiment_name,
              config=vars(self.config)
            )
            wandb.run.name = self.id
            wandb.run.save()
        
        start_epoch=0
        if self.config.resume_checkpoint:
            cpt_dict=self.load()
            if cpt_dict:
                start_epoch=cpt_dict['epoch']
                self.model.load_state_dict(cpt_dict['model_state_dict'])
                self.ema_model.load_state_dict(cpt_dict['ema_model_state_dict'])
                print('Checkpoint has been loaded.')
            else:
                print('Checkpoint has no contents.')
                return
        
        best_perf = [sys.float_info.min]*2
        last_state_dict=[]
        last_acc=0.
        val_loss = AverageMeter()
        
        for epoch in range(start_epoch+1, num_epochs+1):
            self.epoch=epoch
            info_dict=self.trn_iteration(trn_loader, scheduler)
            if self.config.upd_by_ep and scheduler != None: scheduler.step()
        
            if epoch % self.config.save_interval == 0:
                self.save(epoch, [self.model.state_dict(), self.ema_model.state_dict()])
            
            if epoch % self.config.validation_interval == 0:
                info_dict.update(self.val_iteration(val_loader, val_loss))
                
                if info_dict['val_acc'] >= best_perf[1]:
                    best_perf[0]=epoch
                    best_perf[1]=info_dict['val_acc']
                    
                if epoch == num_epochs:
                    last_acc= info_dict['val_acc']
            
            if self.config.use_wandb:
                    wandb.log(info_dict)
            
            logger.logkvs(info_dict)
            logger.dumpkvs()
            str_output = f'epoch & Acc \n'
            str_output += f'{best_perf[0]} & {best_perf[1]:.4f} \n'
            logger.log(str_output)
            
            if val_loss.is_overfitting() and self.config.early_stop:
                print('Detected overfitting signs, stop training.')
                break
            
            if self.epoch % self.config.save_interval != 0 and val_loss.shall_save():
                self.save(epoch-1, last_state_dict)   
            
            val_loss.reset()
                
            last_state_dict = [copy.deepcopy(self.model.state_dict()),copy.deepcopy(self.ema_model.state_dict())]
        if self.config.use_wandb:
            wandb.finish()
            
        return last_acc
            
    def save(self, epoch, model_state):
        if epoch < self.config.skip_epoch:
            return
        mkdir(self.cpt_dir)
        cpt_name=self.id+f'_e{epoch}.pt'
        torch.save({
                'epoch': epoch,
                'model_state_dict': model_state[0],
                'ema_model_state_dict': model_state[1]
                }, os.path.join(self.cpt_dir,cpt_name))
        print(f'model saved in {self.cpt_dir}/{cpt_name}')
        
    def load(self):
        if os.path.exists(self.config.resume_checkpoint):
            _, cpt_name=os.path.split(self.config.resume_checkpoint)
            if '.pt' in cpt_name:
                if self.ssl_name in cpt_name:
                    print(f'loading file at {self.config.resume_checkpoint}')
                    return torch.load(self.config.resume_checkpoint, map_location=self.device)
                else:
                    print(f'Not {self.ssl_name} checkpoint!')
            else:
                print('please provide a file path with suffix of ".pt"')
                return None
        else:
            print(f'{self.config.resume_checkpoint} does not exists!')
            return None

    def update_ema(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step +1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1-alpha)
    
print('MeanTeacher.py')