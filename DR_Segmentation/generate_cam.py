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

def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + ' ----- folder created')
        return True
    else:
        # print(path + ' ----- folder existed')
        return False
    
def read(path):
    img=Image.open(path)
    img=np.array(img)
    return img

def mycrop(raw_img, loc=None):
    pixel_lim =0
    lim=5
    if loc==None:
        img=(np.max(raw_img,axis=2)>0).astype(np.int32)
        # plt.figure()
        # plt.imshow(img)
        # plt.show()
        # plt.close()
        l = None
        r = img.shape[1]
        # print(np.sum(img[:,1] > pixel_lim))
        # print(np.sum(img[:,300] > pixel_lim))
        
        for i in range(img.shape[1]):
            _sum=np.sum(img[:,i] > pixel_lim)
            # _sum = np.max(img[:,i])
            # if _sum >0:
                # print(f'{i}, sum:{_sum}')
                # print(img[:,i])
            if  _sum> lim and l == None:
                l = i
            elif _sum < lim and l:
                r = i
                break
        t = None
        b = img.shape[0]
        for i in range(img.shape[0]):
            _sum=np.sum(img[i] > pixel_lim)
            # _sum = np.max(img[i])
            if _sum > lim and t == None:
                t = i
            elif _sum < lim and t:
                b = i
                break
        loc = [t,b,l,r]

    img = raw_img[loc[0]:loc[1],loc[2]:loc[3]]    
    return img, loc

def draw_masks_fromList(image, masks_generated, label2colors) :
    masked_image = image.copy()
    for i,(l,c) in enumerate(label2colors.items(), start=1):
        masked_image = np.where(masks_generated==i,#np.repeat(masks_generated[i][:, :, np.newaxis], 3, axis=2),
                                np.asarray(c, dtype='uint8'),
                                masked_image)
        
        masked_image = masked_image.astype(np.uint8)
    
    return cv2.addWeighted(image, 0.3, masked_image, 0.7, 0)

def eliminate_overlap(ms,m):
    _m=m>=1
    _ms=ms>=1
    inter=_m==_ms
    # print(inter.reshape(-1).tolist().count(1))
    m=np.where(inter,0,m)
    return m

    

root=os.path.join('D:\PY\PYWORK\datasets\ddr\lesion_segmentation')

trn_image_path = os.path.join(root,'train','image')
val_image_path = os.path.join(root,'valid','image')

trn_mask_path = os.path.join(root, 'train','label')
val_mask_path = os.path.join(root, 'valid','segmentation label')

tasks = {'MA':'MA','HE':'HE','EX':'EX','SE':'SE'}

imgs = os.listdir(root)
target = 'D:\PY\PYWORK\datasets\ddr\cropped_clahe28_ddr'
paths = {
        'trn':
            {
                'image_root':trn_image_path,
                 'mask_root':trn_mask_path
            },
         'val':
            {
                'image_root':val_image_path,
                 'mask_root':val_mask_path
            }
        }


mkdir(target)
clahe=CLAHE(clip_limit=2,tile_grid_size=(8,8))

label2colors={'MA':(255,0,0),'HE':(0,0,255),'EX':(255,0,255),'SE':(0,255,0)}

for i,(k,v) in enumerate(paths.items()):
    t = os.path.join(target, k)
    image_target=os.path.join(t, 'image')
    mkdir(image_target)
    image_root=v['image_root']
    mask_root=v['mask_root']
    images = os.listdir(image_root)

    with tqdm(total=len(images), desc=f"croping {k} set", unit='images') as pbar:
        for num,img in enumerate(images, start=1):
            n=img
            img=read(os.path.join(image_root,n))
            o_img=img
            # print(f'image loc:{loc}')
            # print(f'imgae right:{np.max(o_img[:,loc[2]-1])}, left:{np.max(o_img[:,loc[3]+1])}, top:{np.max(o_img[:,loc[0]+1])}, bottom:{np.max(o_img[:,loc[1]-1])}')
            print(img.shape)
            print(n)
            print(f'min:{np.min(img)}, max:{np.max(img)}')
            img=clahe(image=img)['image']
            img, loc=mycrop(img)
            print(img.shape)
            
            cv2.imwrite(os.path.join(image_target,n),img[:,:,::-1])
            
            idx=n.split('.')[0]
            mask=np.zeros(img.shape[0:2],dtype=np.int32)
            for i,task in enumerate(tasks.items()):
                suffix = '.tif'
                mask_name = idx + suffix  # if idx = 0. we look for the image 1
                mask_path = os.path.join(mask_root, task[1], mask_name)
                if os.path.exists(mask_path):
                    m = read(mask_path)
                    m,_ = mycrop(m,loc)
                    mask_target = os.path.join(t,'GT', task[0])
                    mkdir(mask_target)
                    if len(m.shape)==3:
                        m=m[:,:,0]/765
                        _m=Image.fromarray(m)
                        _m.save(os.path.join(mask_target, mask_name), format='TIFF')
                    else:
                        cv2.imwrite(os.path.join(mask_target, mask_name), m)
                    m = m.astype(np.int32)
                    m = m*(i+1)
                    m = eliminate_overlap(mask,m)
                    mask+=m
            # if img.shape[0]<100 or img.shape[1]<100:
            # print(f'image loc:{loc}')
            # print(f'imgae right:{np.max(o_img[:,loc[2]])}, left:{np.max(o_img[:,loc[3]])}, top:{np.max(o_img[:,loc[0]])}, bottom:{np.max(o_img[:,loc[1]])}')
        
            # mask=np.expand_dims(mask,axis=2)
            # image_with_mask=draw_masks_fromList(img, mask, label2colors)
            
            # imgs=[o_img, img, image_with_mask]
            # titles=[f'{num}th image: {n}', 'clahe_2_8', 'mask']
            # patches=[mpatches.Patch(color=np.array(c)/255, label=l) for l,c in label2colors.items()]
            # fig,axs = plt.subplots(1,len(imgs),figsize=(15,45),sharey=True)
            # for i,ax in enumerate(axs):
            #     ax.imshow(imgs[i])
            #     ax.set_title(titles[i])
            #     if titles[i]=='mask':
            #         ax.legend(bbox_to_anchor=[0.82,1],loc='upper left',handles=patches)
            #     ax.axis('off')
            # plt.show()
            pbar.update(1)
            break
    