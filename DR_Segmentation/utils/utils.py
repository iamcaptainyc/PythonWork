import sys, os
import argparse
import csv

import numpy as np
import torch
import itertools
import json
from torch.utils.data.sampler import Sampler

NO_LABEL=-1

# from const import *


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)
        
class Aobj():
    def __init__(self,default_dict):
        for k, v in default_dict.items():
            setattr(self,k,v)
    def get_dict(self):
        return self.__dict__
    
    def update(self, default_dict):
        for k, v in default_dict.items():
            setattr(self,k,v)


def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + ' ----- folder created')
        return True
    else:
        return False

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
        self.warning=0
        self.min_avg=sys.float_info.max

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def is_overfitting(self):
        if self.min_avg < self.avg:
            print(f'warning val_loss:{self.avg} > minimum val_loss:{self.min_avg}')
            self.warning+=1
            print(f'warning count: {self.warning}')
        elif self.min_avg > self.avg:
            self.warning=0
            self.min_avg=self.avg
        return self.warning>=3
    
    def shall_save(self):
        return self.warning==1
    
def count_class(label, num_classes):
    if not isinstance(num_classes, list):
        num_classes=[num_classes]
        label=np.expand_dims(label, axis=0)
    if label.shape[0] != len(num_classes):
        label=label.transpose()
    results={}
    for i,n in enumerate(num_classes):
        l=label[i].tolist()
        result={}
        for j in range(n):
            result[f'第{j}类']=l.count(j)
        results.update({f'标签{i}':result})
    # return json.dumps(results, indent=4, ensure_ascii=False)
    return results

print('utils.py')