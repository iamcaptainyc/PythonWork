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

def crop(img):
    crop_x_left=260
    crop_x_right=3690
    img=img[:,crop_x_left:crop_x_right]    
    return img


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

    

root=os.path.join(r'D:\PY\PYWORK\datasets\IDRID\A. Segmentation')
image_root = os.path.join(root, '1. Original Images')
mask_root = os.path.join(root, '2. All Segmentation Groundtruths')

trn_image_path = os.path.join(image_root,'a. Training Set')
val_image_path = os.path.join(image_root,'b. Testing Set')

trn_mask_path = os.path.join(mask_root, 'a. Training Set')
val_mask_path = os.path.join(mask_root, 'b. Testing Set')

tasks = {'MA':'1. Microaneurysms','HE':'2. Haemorrhages','EX':'3. Hard Exudates','SE':'4. Soft Exudates'}

imgs = os.listdir(root)
target = ''
paths = {
         'val':
            {'image_root':val_image_path,
             'mask_root':val_mask_path
            }
        }
# 'trn':
#             {'image_root':trn_image_path,
#              'mask_root':trn_mask_path
#             },

# mkdir(target)
clahe=CLAHE(clip_limit=2,tile_grid_size=(8,8))

label2colors={'MA':(255,0,0),'HE':(0,0,255),'EX':(255,0,255),'SE':(0,255,0)}

for i,(k,v) in enumerate(paths.items()):
    t = os.path.join(target, k)
    image_target=os.path.join(t, 'image')
    # mkdir(image_target)
    image_root=v['image_root']
    mask_root=v['mask_root']
    images = os.listdir(image_root)
    
    for num,img in enumerate(images, start=1):
        n=img
        img=read(os.path.join(image_root,n))
        o_img=img
        img=clahe(image=img)['image']
        # img=crop(img)
        # cv2.imwrite(os.path.join(image_target,n),img[:,:,::-1])
        print(n)
        
        idx=n.split('_')[1][:2]
        mask=np.zeros(img.shape[0:2],dtype=np.int32)
        for i,task in enumerate(tasks.items()):
            suffix = '.tif'
            mask_name = 'IDRiD_'+ idx + '_' + task[0] + suffix  # if idx = 0. we look for the image 1
            mask_path = os.path.join(mask_root, task[1], mask_name)
            if os.path.exists(mask_path):
                m = read(mask_path)
                # m = crop(m)
                # mask_target = os.path.join(t,'GT', task[0])
                # mkdir(mask_target)
                # if len(m.shape)==3:
                #     m=m[:,:,0]/765
                #     _m=Image.fromarray(m)
                #     _m.save(os.path.join(mask_target, mask_name), format='TIFF')
                # else:
                #     cv2.imwrite(os.path.join(mask_target, mask_name), m)
                m = m.astype(np.int32)
                m = m*(i+1)
                m = eliminate_overlap(mask,m)
                mask+=m
        mask=np.expand_dims(mask,axis=2)
        image_with_mask=draw_masks_fromList(img, mask, label2colors)
        
        temp_save_path = os.path.join('D:\PY')
        cv2.imwrite(os.path.join(temp_save_path, f'{n.split(".")[0]}_mask.jpg'), image_with_mask[1630:2170, 2260:3140,::-1])
        
        cv2.imwrite(os.path.join(temp_save_path, f'{n.split(".")[0]}_crop.jpg'), o_img[1630:2170, 2260:3140,::-1])
        
        cv2.imwrite(os.path.join(temp_save_path, f'{n}'), o_img[:,:,::-1])
        
        imgs=[o_img, img, image_with_mask]
        titles=[f'{num}th image: {n}', 'clahe_2_8', 'mask']
        patches=[mpatches.Patch(color=np.array(c)/255, label=l) for l,c in label2colors.items()]
        fig,axs = plt.subplots(1,len(imgs),figsize=(15,45),sharey=True)
        for i,ax in enumerate(axs):
            ax.imshow(imgs[i])
            ax.set_title(titles[i])
            if titles[i]=='mask':
                ax.legend(bbox_to_anchor=[0.82,1],loc='upper left',handles=patches)
            ax.axis('off')
        plt.show()
        
        if num==1:
            break
    