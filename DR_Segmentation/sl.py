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

class SLTrainer:
    def __init__(self, model, optimizer, criterion, device, config):
        print("Supervised trainer")
        self.ssl_name='SL'
        self.config=config
        self.model=model
        self.optimizer=optimizer
        self.device=device
        self.criterion=criterion
        self.estimator=Estimator(config.metrics, config.num_classes)
        self.id=config.id
        
        if self.config.dual and self.config.seg_vessel:
            self.criterion_extra=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(10).to(self.device))
        
        self.data_dir=os.path.join(config.checkpoint_dir,'Supervised', self.id)
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
        self.estimator.reset()
        trn_loss = AverageMeter()
        
        lr=self.optimizer.state_dict()['param_groups'][0]['lr']
        with tqdm(total=len(data_loader), desc=f"Epoch {self.epoch}/{self.config.num_epochs} --- training lr:{lr}", unit='batch') as pbar:
            for step, (x, y) in enumerate(data_loader):
                if self.config.dual:
                    x = x.to(self.device)
                    x = Variable(x)
                    y = [_y.to(self.device) for _y in y]
                    y_extra = Variable(y[1]).float()
                    y = Variable(y[0]).float()
                    if self.config.seg_vessel:
                        y_pred = self.model(x.float())
                        y_extra_pred = y_pred[1].squeeze()
                        y_pred = y_pred[0]
                    else:
                        y_pred = self.model(x.float(), y_extra)
                else:
                    x, y = [t.to(self.device) for t in (x,y)]
                    x, y=Variable(x),Variable(y)
                    y_pred = self.model(x.float())
                    
                loss = self.criterion(y_pred, y.long())
                if self.config.dual and self.config.seg_vessel:
                    loss_extra = self.criterion_extra(y_extra_pred, y_extra)
                    loss += self.config.dual_loss_factor*loss_extra
                
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
                self.estimator.update(y_pred, y)
                pbar.update(1)
        
        score=self.estimator.get_scores(4)

        for m in set(self.estimator.need_named_metrics)&set(score.keys()):
            val=score.pop(m)
            score.update(self.estimator.name_val(data_loader.dataset.tasks, m, val))

        if self.config.show_log:
            print(f'trn_gt:{self.estimator.count("gt")}')
            print(f'trn_pred:{self.estimator.count("pred")}')
        
        info_dict=dict([('epoch', self.epoch),
                   ('lr', lr),
                   ('trn_loss', round(trn_loss.avg,4))])
        
        info_dict.update({'trn_'+k:v for k,v in score.items()})
        # if self.config.dual:
        #     k=self.config.dual
        #     score=self.estimator_extra.get_scores(4)
        #     if self.config.show_log:
        #         print(f'trn_gt_{k}:{self.estimator_extra.count("gt")}')
        #         print(f'trn_pred_{k}:{self.estimator_extra.count("pred")}')
        #     info_dict.update({'trn_'+_k+'_'+k:_v for _k,_v in score.items()})
        return info_dict
        
    def val_iteration(self, data_loader, val_loss):
        self.model.eval()
        
        self.estimator.reset()
        
        with tqdm(total=len(data_loader), desc=f"Epoch {self.epoch}/{self.config.num_epochs} --- validating", unit='batch') as pbar:
            with torch.no_grad():
                for step, (x, y) in enumerate(data_loader):
                    if self.config.dual:
                        x = x.to(self.device)
                        x = Variable(x)
                        y = [_y.to(self.device) for _y in y]
                        y_extra = Variable(y[1]).float()
                        y = Variable(y[0]).float()
                        if self.config.seg_vessel:
                            y_pred = self.model(x.float())
                            y_extra_pred = y_pred[1].squeeze()
                            y_pred = y_pred[0]
                        else:
                            y_pred = self.model(x.float(), y_extra)
                    else:
                        x, y = [t.to(self.device) for t in (x,y)]
                        x, y=Variable(x),Variable(y)
                        y_pred = self.model(x.float())
    
                    loss=self.criterion(y_pred, y.long())
                    if self.config.dual and self.config.seg_vessel:
                        loss_extra = self.criterion_extra(y_extra_pred, y_extra)
                        loss += self.config.dual_loss_factor*loss_extra
                        
                    if self.config.accumulation_step:
                        loss /= self.config.accumulation_step
                        if ((step + 1) % self.config.accumulation_step == 0) or (step + 1 == len(data_loader)):
                            val_loss.update(loss.item()*self.config.accumulation_step)
                    else: 
                        val_loss.update(loss.item())
    
                    self.estimator.update(y_pred, y)
                    pbar.update(1) 
                
        score=self.estimator.get_scores(4)
        
        for m in set(self.estimator.need_named_metrics)&set(score.keys()):
            val=score.pop(m)
            score.update(self.estimator.name_val(data_loader.dataset.tasks, m, val))

        if self.config.show_log:
            print(f'val_gt:{self.estimator.count("gt")}')
            print(f'val_pred:{self.estimator.count("pred")}')
        
        info_dict=dict([('val_loss', round(val_loss.avg,4))])
        info_dict.update({'val_'+k:v for k,v in score.items()})
        # info_dict.update({f'roc_e{self.epoch}':self.plot_roc(self.estimator.y, self.estimator.y_pred)})
        
        return info_dict
    
    def loop(self, num_epochs, trn_loader, val_loader):
        logger.configure(dir=self.log_dir, log_suffix=self.id)
        
        if self.config.use_wandb:
            wandb.init(
              project=self.config.experiment_name,
              config=vars(self.config)
            )
            wandb.run.name = self.id
            wandb.run.save()
            
        self.load()
            
        scheduler=get_scheduler(self.config, self.optimizer, last_epoch = self.last_epoch)
        
        best_perf = [sys.float_info.min]*2
        val_loss = AverageMeter()
        last_info=None

        if self.config.use_wandb:
            for k,v in self.model.named_children():
                print(f'{k}:{v}')
        
        for epoch in range(self.start_epoch+1, num_epochs+1):
            self.epoch=epoch
            info_dict=self.trn_iteration(trn_loader, scheduler)
            if self.config.upd_by_ep and scheduler != None: scheduler.step()
                    
            if epoch % self.config.save_interval == 0:
                self.save(epoch, self.model.state_dict(), self.optimizer.state_dict())
            
            if epoch % self.config.validation_interval == 0 or epoch == 1:
                info_dict.update(self.val_iteration(val_loader, val_loss))
                
                if info_dict['val_'+self.config.best_metric] > best_perf[1]:
                    best_perf[0]=epoch
                    best_perf[1]=info_dict['val_'+self.config.best_metric]
                    self.best_checkpoint = [os.path.join(self.cpt_dir, self.id+f'_e{epoch}.pt')]
                    if epoch != num_epochs:
                        self.best_checkpoint.append(os.path.join(self.cpt_dir, self.id+f'_e{num_epochs}.pt'))
                    if epoch % self.config.save_interval != 0 and info_dict['val_'+self.config.best_metric] > self.config.good_value:
                        self.save(epoch, self.model.state_dict(), self.optimizer.state_dict())
                    
                if epoch == num_epochs:
                    last_info= {self.config.best_metric:info_dict['val_'+self.config.best_metric],'best_cpt':self.best_checkpoint}
                    
            if self.config.use_wandb:
                    wandb.log(info_dict)

            if self.config.show_log:
                logger.logkvs(info_dict)
                logger.dumpkvs()
                str_output = 'epoch & {} \n'.format(self.config.best_metric)
                str_output += f'{best_perf[0]} & {best_perf[1]:.4f} \n'
                logger.log(str_output)
            
            if val_loss.is_overfitting() and self.config.early_stop:
                print('Detected overfitting signs, stop training.')
                break
            
            val_loss.reset()
                
        if self.config.use_wandb:
            wandb.finish()
            
        return last_info
    
    def evaluate(self, val_loader):
        print('-------------------EVALUATING-------------------')
        
        logger.configure(dir=self.log_dir, log_suffix=self.id)
        
        if self.config.use_wandb:
            wandb.init(
              project=self.config.experiment_name,
              config=vars(self.config)
            )
            wandb.run.name = self.id
            wandb.run.save()
        
        self.load()
        
        val_loss = AverageMeter()
        self.epoch=1
        info_dict=self.val_iteration(val_loader, val_loss)
        
        if self.config.use_wandb:
            wandb.log(info_dict)
        logger.logkvs(info_dict)
        logger.dumpkvs()
        
    def inference(self, data_loader, type):
        print('-------------------INFERENCE-------------------')
        logger.configure(dir=self.log_dir, log_suffix=self.id)
        
        self.load()
        
        self.model.eval()
        
        result_dir=os.path.join(self.config.inference_root_dir, self.config.dataset_name, type)
        mkdir(result_dir)
        
        with tqdm(total=len(data_loader), desc=f"Inference", unit='batch') as pbar:
            with torch.no_grad():
                for step, (n, x, y) in enumerate(data_loader):
                    x, y = x.to(self.device), y.to(self.device)
                    x, y = Variable(x), Variable(y)
    
                    y_pred = self.model(x.float())
                    logits = y_pred.detach().cpu()
                    x_np = x.detach().cpu().numpy()
                    
                    logits = F.softmax(logits, dim=1)
                    predict = np.argmax(logits.numpy(), axis=1).astype(np.uint8)
                    
                    for i,p in enumerate(predict):
                        img = Image.fromarray(p)
                        result_path=os.path.join(result_dir, n[i]+'_'+self.config.inference_suffix+'.tif')
                        print(result_path)
                        img.save(result_path, format='TIFF')
                        
                        o_img=np.transpose(x_np[i],(1,2,0))
                        imgs=[o_img, img]
                        titles=[n[i], 'vessel_mask']
                        fig,axs = plt.subplots(1,len(imgs),figsize=(15,30),sharey=True)
                        for i,ax in enumerate(axs):
                            ax.imshow(imgs[i])
                            ax.set_title(titles[i])
                            ax.axis('off')
                        plt.show()
                    
                    pbar.update(1) 
        
    def plot_roc(self, y, y_pred):
        fpr, tpr, thresholds = roc_curve(y, y_pred)
        roc_auc = auc(fpr, tpr)
        fig=plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic e{self.epoch}')
        plt.legend(loc="lower right")
        import matplotlib_inline
        matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
        plt.show()
        plt.save(f'/kaggle/working/roc_{self.epoch}.svg')
        plt.close()
        return fig
    
    def plot_cm(self,cm,name):
        import seaborn as sns
        plt.figure()
        sns.heatmap(cm, annot=True, cmap="Blues", cbar=False, fmt="2d")

        # 设置标签和标题
        plt.xlabel('Predict')
        plt.ylabel("Ground Truth")
        plt.gca().xaxis.tick_top()
        plt.gca().xaxis.set_label_position('top')
        # plt.title()
#         import matplotlib_inline
#         matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
        plt.savefig(os.path.join(self.data_dir,f'{name}_cm_e{self.epoch}.svg'), format='svg')
#         # 显示图像
#         plt.show()
        plt.close()
        
    def save(self, epoch, model_state, optim_state, name=None):
        if epoch < self.config.skip_epoch:
            return
        mkdir(self.cpt_dir)
        if name != None:
            cpt_name=name
        else: cpt_name=self.id+f'_e{epoch}.pt'
        save_dict={'epoch': epoch, 'args':self.config, 'model_state_dict': model_state}
        if self.config.save_optim:
            save_dict.update({'optim_state_dict': optim_state})
        torch.save(save_dict, os.path.join(self.cpt_dir,cpt_name))
        print(f'model saved in {self.cpt_dir}/{cpt_name}')
        
    def load(self):
        if self.config.resume_checkpoint:
            if os.path.exists(self.config.resume_checkpoint):
                _, cpt_name=os.path.split(self.config.resume_checkpoint)
                if '.pt' in cpt_name:
                    print(f'loading file at {self.config.resume_checkpoint}')
                    cpt_dict = torch.load(self.config.resume_checkpoint, map_location=self.device)
                else:
                    print('please provide a file path with suffix of ".pt"')
                    return
            else:
                print(f'{self.config.resume_checkpoint} does not exists!')
                return
            
            if cpt_dict:
                self.start_epoch=cpt_dict['epoch']
                self.model.load_state_dict(cpt_dict['model_state_dict'])
                if self.config.load_optimizer:
                    op_state=cpt_dict.get('optim_state_dict', -1)
                    if op_state !=-1:
                        self.last_epoch=self.start_epoch-1
                        self.optimizer.load_state_dict(op_state)
                    else:
                        self.last_epoch=-1
                print('Checkpoint has been loaded.')
            else:
                print('Checkpoint has no contents.')
                return

print('sl.py')