import torch
import torcheval.metrics as tm
import numpy as np
from sklearn.metrics import *
import matplotlib.pyplot as plt
from torch.nn import functional as F

# from utils import utils
# from const import *


class Estimator():
    def __init__(self, metrics, num_classes, binary=False, input_logits=True):
        
        self.binary=binary
        self.input_logits=input_logits
        self.threshold=0.5
        if self.binary:
            self.metrics_names = {
                'acc': tm.BinaryAccuracy,
                'f1': tm.BinaryF1Score,
                'auc': tm.BinaryAUROC,
                'auprc': tm.BinaryAUPRC,
                'precision': tm.BinaryPrecision,
                'recall': tm.BinaryRecall,
                'cm':tm.BinaryConfusionMatrix
            }
        else:
            self.metrics_names = {
                'acc': tm.MulticlassAccuracy,
                'f1': tm.MulticlassF1Score,
                'auc': tm.MulticlassAUROC,
                'auprc': tm.MulticlassAUPRC,
                'precision': tm.MulticlassPrecision,
                'recall': tm.MulticlassRecall,
                'kappa': QuadraticWeightedKappa,
                'SegMetrics': SegMetrics
            }
        
        self.return_dict_metrics = ['SegMetrics']
        self.logits_required_metrics = ['auc', 'auprc']
        self.need_kwargs_change=['f1','precision','recall']
        self.need_named_metrics=['auc', 'auprc', 'PA']
        
        
        self.num_classes = num_classes
        self.metrics = metrics
        self.metrics_fn={}
        
        if self.binary:
            for m in metrics:
                self.metrics_fn[m]=self.metrics_names[m]()
        else:
            for m in metrics:
                if m in self.need_kwargs_change:
                    self.metrics_fn[m]=self.metrics_names[m](num_classes=num_classes, average='macro')
                if m in self.logits_required_metrics:
                    self.metrics_fn[m]=self.metrics_names[m](num_classes=num_classes, average=None)
                else:
                    self.metrics_fn[m]=self.metrics_names[m](num_classes=num_classes)
                
        # self.y = np.empty(0)
        # self.y_pred = np.empty(0)
        # if self.binary:
        #     self.logits = np.empty(0)
        # else:
        #     self.logits = np.empty((0,self.num_classes))
            

    def update(self, predictions, targets):
        targets = targets.detach().cpu().long()
        logits = predictions.detach().cpu()
        predictions = self.to_prediction(logits)
        
        if len(logits.shape) == 4:
            logits = logits.permute(0,2,3,1).reshape(-1, self.num_classes)
            logits = F.softmax(logits, dim=1)
        elif 'kappa' in self.metrics:
            logits = F.softmax(logits, dim=1)
        else:
            if self.input_logits:
                logits = F.sigmoid(logits)
            logits = logits.contiguous().view(-1)
        
            
        predictions = predictions.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        
        # self.y_pred = np.concatenate((self.y_pred, predictions.numpy()), axis=0)
        # self.y = np.concatenate((self.y, targets.numpy()), axis=0)
        # self.logits = np.concatenate((self.logits, logits.numpy()), axis=0)

        # update metrics
        for m in self.metrics_fn.keys():
            if m in self.logits_required_metrics:
                self.metrics_fn[m].update(logits, targets)
            else:
                self.metrics_fn[m].update(predictions, targets)

    def get_scores(self, digits=-1):
        
#         for i in range(self.num_classes):
#             self.plot_auprc(self.y, self.logits, i)
        
        scores={}
        for m in self.metrics:
            scores.update(self._compute(m, digits))
        return scores

    # def count(self, name):
    #     if name == 'gt' :
    #         return count_class(self.y, self.num_classes)
    #     elif name == 'pred': 
    #         return count_class(self.y_pred, self.num_classes)

    def _compute(self, metric, digits=-1):
        if not self.binary:
            if metric in self.return_dict_metrics:
                return self.metrics_fn[metric].compute(digits)
            if metric in self.logits_required_metrics:
                l=self.metrics_fn[metric].compute().tolist()
                l.append(np.mean(l[1:]))
                l=np.round(l,digits)
                return {metric:l}
            
        score = self.metrics_fn[metric].compute().item()
        score = score if digits == -1 else round(score, digits)
        return {metric:score}

    def reset(self):
        for m in self.metrics_fn.keys():
            self.metrics_fn[m].reset()
        # self.y = np.empty(0)
        # self.y_pred = np.empty(0)
        # if self.binary:
        #     self.logits = np.empty(0)
        # else:
        #     self.logits = np.empty((0,self.num_classes))
    
    def to_prediction(self, predictions):
        if self.binary:
            predictions = (predictions>self.threshold).to(torch.int16)
        else:
            predictions = torch.argmax(predictions, dim=1).long()
        return predictions
    
    def name_val(self, label, metric, val):
        l=label
        label={'BK':None}
        label.update(l)
        named_val={}
        for i,(k,v) in enumerate(label.items()):
            named_val.update({metric+'_'+k:val[i]})
        named_val.update({metric:val[-1]})
        return named_val
    
    def plot_auprc(self, targets, logits, c):
        targets=(targets==c).astype(np.int32)
        logits=logits[:,c]
        PrecisionRecallDisplay.from_predictions(targets ,logits)
        precision, recall, th = precision_recall_curve(targets, logits)
        print('precision:{}\n'.format(precision))
        print('recall:{}\n'.format(recall))
        print('threshhold:{}\n'.format(th))
        auprc=auc(recall, precision)
        
        plt.legend([f'AUPRC = {auprc:.2f}'], loc="lower left")
        plt.title(f'{c}th PR Curve')
        plt.show()

class BinaryAUPRC():
    def __init__(self):
        self.logits = np.empty(0)
        self.targets = np.empty(0)
        
    def update(self, logits, targets):
        self.targets = np.concatenate((self.targets, targets.numpy()), axis=0)
        self.logits = np.concatenate((self.logits, logits.numpy()), axis=0)
        
    def compute(self, digits=4):
        precision, recall, thresholds = precision_recall_curve(self.targets, self.logits)
        auprc = auc(recall, precision)
        return np.round(auprc, digits)
        
    def reset(self):
        self.logits = np.empty(0)
        self.targets = np.empty(0)
            
class SegMetrics():
    def __init__(self, num_classes):
        self.num_classes=num_classes
        self.confusion_matrix = np.zeros((num_classes,num_classes))
        self.cm=tm.MulticlassConfusionMatrix(num_classes=num_classes)
        
    def PA(self):
        return np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
    
    def PAC(self):
        acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        acc_class = np.nanmean(acc)
        return acc_class
    
    def MIoU(self):
        IoU = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))
        MIoU = np.nanmean(IoU)
        MIoU_noback = np.nanmean(IoU[1:])
        return MIoU, MIoU_noback
    
    def update(self, predictions, targets):
        self.cm.update(predictions, targets)
        
    def reset(self):
        self.cm = tm.MulticlassConfusionMatrix(num_classes=self.num_classes)

    def compute(self, digits):
        self.confusion_matrix=self.cm.compute().numpy().astype(np.int32)
        cm=self.cm.normalized('true').numpy()*100
        np.set_printoptions(suppress=True)
        print(np.round(cm,2))
        MIoU, MIoU_noback = self.MIoU()
        return {
            "PA": np.round(self.PA(), digits),
            "PAC": np.round(self.PAC(), digits),
            "MIoU": np.round(MIoU, digits),
            "MIoU_NoBack": np.round(MIoU_noback, digits)
        }

print('metircs.py')
