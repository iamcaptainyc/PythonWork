import torch
import torch.nn as nn
import os
from PIL import Image
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from albumentations.augmentations.transforms import CLAHE
import cv2
from tqdm import tqdm

def mycrop(raw_img, loc=None):
    pixel_lim =0
    lim=0
    if loc==None:
        print(np.min(raw_img[:,:,0]))
        img=(raw_img[:,:,0]>0).astype(np.int32)
        print(np.min(img))
        plt.figure()
        plt.imshow(img)
        plt.show()
        plt.close()
        l = None
        r = img.shape[1]
        # print(np.sum(img[:,1] > pixel_lim))
        # print(np.sum(img[:,300] > pixel_lim))
        
        for i in range(img.shape[1]):
            _sum=np.max(img[:,i])
            if  _sum> lim:
                l = i
                break
        for i in range(img.shape[1]-1,l,-1):
            _sum=np.max(img[:,i])
            if _sum > lim:
                r = i
                break
        t = None
        b = img.shape[0]
        for i in range(img.shape[0]):
            _sum = np.max(img[i])
            if _sum > lim:
                t = i
                break
        for i in range(img.shape[0]-1,t,-1):
            _sum = np.max(img[i])
            if _sum > lim:
                b = i
                break
        loc = [t,b,l,r]

    img = raw_img[loc[0]:loc[1],loc[2]:loc[3]]    
    return img, loc

def read(path):
    img=Image.open(path)
    img=np.array(img)
    return img

path =os.path.join(r'D:\PY\PYWORK\datasets\ddr\lesion_segmentation\test\image\007-3820-200.jpg')
img=read(path)
plt.figure()
plt.imshow(img)
plt.show()