import torch
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt


class IDRiDDataset(Dataset):
    def __init__(self, path, dataset_type='train', resolution=[512,700], leisions=['MA','HE','EX','SE'], transform=None, args=None):
        super(IDRiDDataset, self).__init__()
        self.task_dicts = {'MA':'1. Microaneurysms','HE':'2. Haemorrhages','EX':'3. Hard Exudates','SE':'4. Soft Exudates','VES':'5. Vessel/640'}
        self.tasks=self.get_tasks(leisions)
        self.label2color = {'MA':[0,0,255],'HE':[0,255,0],'EX':[255,0,0],'SE':[255,0,255]}

        self.preprocess=args.preprocess
        self.resolution=resolution
        self.dual=args.dual
        self.args=args
        
        
        if self.preprocess:
            preprocess_path = os.path.join(path, 'clahe', '2_8_cropped')
            if dataset_type == 'train':
                self.image_root=os.path.join(preprocess_path, 'trn', 'image')
                self.mask_root=os.path.join(preprocess_path, 'trn', 'GT')
            elif dataset_type == 'val':
                self.image_root=os.path.join(preprocess_path, 'val', 'image')
                self.mask_root=os.path.join(preprocess_path, 'val', 'GT')
            else:
                raise EnvironmentError('You should put a valid mode to generate the dataset')
        else:
            if dataset_type == 'train':
                self.image_root=os.path.join(path,'1. Original Images','a. Training Set')
                self.mask_root=os.path.join(path,'2. All Segmentation Groundtruths','a. Training Set')
            elif dataset_type == 'val':
                self.image_root=os.path.join(path,'1. Original Images', 'b. Testing Set')
                self.mask_root=os.path.join(path,'2. All Segmentation Groundtruths','b. Testing Set')
            else:
                raise EnvironmentError('You should put a valid mode to generate the dataset')
        self.image_names=os.listdir(self.image_root)
        self.dataset_type=dataset_type
        self.transform = transform
        print('{} dataset contains {} images at {}'.format(self.dataset_type, len(self.image_names), self.image_root))
        print(self.image_names)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        'Generate one batch of data'
        sample = self.load(idx)
        
        if self.args.inference:
            image_name=self.image_names[idx].split('.')[0]
            return image_name, sample['image'], sample['masks']
        if self.args.dual:
            return sample['image'], sample['masks']
        return sample['image'], sample['masks'][0]
    
    def get_tasks(self, leisions):
        tasks={}
        for k,v in self.task_dicts.items():
            if k in leisions:
                tasks.update({k:v})
        return tasks

    def eliminate_overlap(self,ms,m):
        # print('masks : {}  m : {}'.format(ms.shape, m.shape))
        _m=m>=1
        _ms=ms>=1
        inter=np.logical_and(_ms,_m)
        # print('inter:{}'.format(inter.reshape(-1).tolist().count(True)))
        m=np.where(inter,0,m)
        return m
    
    def read_mask(self,path):
        mask=Image.open(path)
        mask=np.array(mask)
        #这里不用归一化，因为掩码值已经在0-1内
        mask = cv2.resize(mask, self.resolution)
        mask=mask.astype(np.int32)#这里必须是long，否则传入损失函数时会报错
#         print('read mask shape:{}'.format(mask.shape))
        return mask
    
    def read_image(self,path):
        image = Image.open(path)
        image = np.array(image)
        image = cv2.resize(image, self.resolution)
        image = np.transpose(image, (2,0,1))#(512, 512, 3)->(3, 512, 512)
        image = image/255.0 #归一化 (3, 512, 512)
        image = image.astype(np.float32)
#         print('read image np shape:{}'.format(image.shape))
        image = torch.from_numpy(image)
        return image
        
    def load(self, idx):
        # Get masks from a particular idx
        image_name=self.image_names[idx]
        image_path = os.path.join(self.image_root, image_name)
        image = self.read_image(image_path)
        
        idx=image_name.split('_')[1][:2]
        
        mask=np.zeros(image.shape[1:3],dtype=np.int32)
#         out_str='\n'
        for i,task in enumerate(self.tasks.items()):
            suffix = '.tif'
            mask_name = 'IDRiD_'+ idx + '_' + task[0] + suffix  # if idx = 0. we look for the image 1
            mask_path = os.path.join(self.mask_root, task[1], mask_name)
            if os.path.exists(mask_path):
                m = self.read_mask(mask_path)*(i+1)
                if len(m.shape)==3:
                    m=np.transpose(m,(2,0,1))
                    m=m[0]/765
                    m=m.astype(np.int32)
                m = self.eliminate_overlap(mask,m)
                mask+=m
#                 out_str+='{} - {} - {} mask : {} | masks : {}\n'.format(image_name, mask_name, task[0], np.unique(m), np.unique(mask))
        
#         print(out_str)
        
        if self.dual:
            suffix='.tif'
            ves_mask_name = 'IDRiD_'+ idx + '_VES' + suffix
            ves_mask_path = os.path.join(self.mask_root, self.task_dicts['VES'], ves_mask_name)
            ves_mask = self.read_mask(ves_mask_path)
            sample = {'image':image, 'masks':[torch.from_numpy(mask),torch.from_numpy(ves_mask)]}
        else:   
            sample = {'image':image , 'masks': [torch.from_numpy(mask)]}
        # If transform apply transformation
        if self.transform:
            sample = self.transform(sample)
        return sample
    
class DDRDataset(Dataset):
    def __init__(self, path, dataset_type='train', resolution=[512,700], leisions=['MA','HE','EX','SE'], transform=None, args=None):
        super(DDRDataset, self).__init__()
        self.task_dicts = {'MA':'MA','HE':'HE','EX':'EX','SE':'SE'}
        self.tasks=self.get_tasks(leisions)
        self.label2color = {'MA':[0,0,255],'HE':[0,255,0],'EX':[255,0,0],'SE':[255,0,255]}

        self.preprocess=args.preprocess
        self.resolution=resolution
        self.dual=args.dual
        self.args=args

        if self.preprocess:
            if dataset_type == 'train':
                self.image_root=os.path.join(path, 'trn', 'image')
                self.mask_root=os.path.join(path, 'trn', 'GT')
            elif dataset_type == 'val':
                self.image_root=os.path.join(path, 'val', 'image')
                self.mask_root=os.path.join(path, 'val', 'GT')
            elif dataset_type == 'test':
                self.image_root=os.path.join(path, 'tst', 'image')
                self.mask_root=os.path.join(path, 'tst', 'GT')
                
            else:
                raise EnvironmentError('You should put a valid mode to generate the dataset')
        else:
            if dataset_type == 'train':
                self.image_root=os.path.join(path, 'train', 'image')
                self.mask_root=os.path.join(path, 'train', 'label')
            elif dataset_type == 'val':
                self.image_root=os.path.join(path, 'valid', 'image')
                self.mask_root=os.path.join(path, 'valid', 'segmentation label')
            elif dataset_type == 'test':
                self.image_root=os.path.join(path, 'test', 'image')
                self.mask_root=os.path.join(path, 'test', 'label')
            else:
                raise EnvironmentError('You should put a valid mode to generate the dataset')
        
            
        
        self.image_names=os.listdir(self.image_root)
        self.dataset_type=dataset_type
        self.transform = transform
        print('{} dataset contains {} images at {}'.format(self.dataset_type, len(self.image_names), self.image_root))
        
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        'Generate one batch of data'
        sample = self.load(idx)
        
        if self.args.inference:
            image_name=self.image_names[idx].split('.')[0]
            return image_name, sample['image'], sample['masks']
        if self.args.dual:
            return sample['image'], sample['masks']
        return sample['image'], sample['masks'][0]
    
    def get_tasks(self, leisions):
        tasks={}
        for k,v in self.task_dicts.items():
            if k in leisions:
                tasks.update({k:v})
        return tasks

    def eliminate_overlap(self,ms,m):
        # print('masks : {}  m : {}'.format(ms.shape, m.shape))
        _m=m>=1
        _ms=ms>=1
        inter=np.logical_and(_ms,_m)
        # print('inter:{}'.format(inter.reshape(-1).tolist().count(True)))
        m=np.where(inter,0,m)
        return m
    
    def read_mask(self,path):
        mask=Image.open(path)
        mask=np.array(mask)
        #这里不用归一化，因为掩码值已经在0-1内
        mask = cv2.resize(mask, self.resolution)
        if not self.preprocess:
            mask = mask/255.0
        mask=mask.astype(np.int32)#这里必须是long，否则传入损失函数时会报错
#         print('read mask shape:{}'.format(mask.shape))
        return mask
    
    def read_image(self,path):
        image = Image.open(path)
        image = np.array(image)
        image = cv2.resize(image, self.resolution)
        image = np.transpose(image, (2,0,1))#(512, 512, 3)->(3, 512, 512)
        image = image/255.0 #归一化 (3, 512, 512)
        image = image.astype(np.float32)
#         print('read image np shape:{}'.format(image.shape))
        image = torch.from_numpy(image)
        return image
        
    def load(self, idx):
        # Get masks from a particular idx
        image_name=self.image_names[idx]
        image_path = os.path.join(self.image_root, image_name)
        image = self.read_image(image_path)
        
        idx=image_name.split('.')[0]
        
        mask=np.zeros(image.shape[1:3],dtype=np.int32)
#         out_str='\n'
        for i,task in enumerate(self.tasks.items()):
            suffix = '.tif'
            mask_name = idx + suffix  # if idx = 0. we look for the image 1
            mask_path = os.path.join(self.mask_root, task[1], mask_name)
            if os.path.exists(mask_path):
                m = self.read_mask(mask_path)*(i+1)
                if len(m.shape)==3:
                    m=np.transpose(m,(2,0,1))
                    m=m[0]/765
                    m=m.astype(np.int32)
                m = self.eliminate_overlap(mask,m)
                mask+=m
#                 out_str+='{} - {} - {} mask : {} | masks : {}\n'.format(image_name, mask_name, task[0], np.unique(m), np.unique(mask))
        
        sample = {'image':image , 'masks': [torch.from_numpy(mask)]}
        # If transform apply transformation
        if self.transform:
            sample = self.transform(sample)
        return sample
    
class VesselDataset(Dataset):
    def __init__(self, path, dataset_type='train', resolution=[512,700], transform=None, args=None):
        super(VesselDataset, self).__init__()
        self.preprocess=args.preprocess
        self.resolution=resolution
        self.args=args
        self.tasks={'VES':'Vessel'}
        
        if self.preprocess:
            preprocess_path = os.path.join(path, 'clahe', '2_8')
            if dataset_type == 'train':
                self.image_root=os.path.join(preprocess_path, 'trn', 'image')
                self.mask_root=os.path.join(preprocess_path, 'trn', 'GT')
            elif dataset_type == 'val':
                self.image_root=os.path.join(preprocess_path, 'val', 'image')
                self.mask_root=os.path.join(preprocess_path, 'val', 'GT')
            else:
                raise EnvironmentError('You should put a valid mode to generate the dataset')
        else:
            if dataset_type == 'train':
                self.image_root=os.path.join(path,'Training','Images')
                self.mask_root=os.path.join(path,'Training','Masks')
            elif dataset_type == 'val':
                self.image_root=os.path.join(path,'Test','Images')
                self.mask_root=os.path.join(path,'Test','Masks')
            else:
                raise EnvironmentError('You should put a valid mode to generate the dataset')
        self.image_names=os.listdir(self.image_root)
        self.dataset_type=dataset_type
        self.transform = transform
        print('{} dataset contains {} images'.format(self.dataset_type, len(self.image_names)))
        print(self.image_names)
        

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        'Generate one batch of data'
        sample = self.load(idx)
        return sample['image'], sample['mask']

    def read_mask(self,path):
        mask=Image.open(path)
        mask=np.array(mask)
        mask=mask/np.max(mask)
        mask = cv2.resize(mask, self.resolution)
        mask=mask.astype(np.int32)#这里必须是long，否则传入损失函数时会报错
#         print('read mask shape:{}'.format(mask.shape))
        return mask
    
    def read_image(self,path):
        image = Image.open(path)
        image = np.array(image)
        image = cv2.resize(image, self.resolution)
        image = np.transpose(image, (2,0,1))#(512, 512, 3)->(3, 512, 512)
        image = image/255.0 #归一化 (3, 512, 512)
        image = image.astype(np.float32)
#         print('read image np shape:{}'.format(image.shape))
        image = torch.from_numpy(image)
        return image
        
    def load(self, idx):
        # Get masks from a particular idx
        image_name=self.image_names[idx]
        image_path = os.path.join(self.image_root, image_name)
        image = self.read_image(image_path)
        
        name=image_name.split('.')[0]

        if name.split('_')[-1] == 'HRF':
            suffix = '.tif'
        elif name.split('_')[-1] == 'DRIVE':
            suffix = '.gif'
        elif name.split('_')[-1] == 'CHASE':
            suffix = '.png'
        mask_name = name + suffix  # if idx = 0. we look for the image 1
        mask_path = os.path.join(self.mask_root, mask_name)
        if os.path.exists(mask_path):
            mask = self.read_mask(mask_path)
#                 out_str+='{} - {} - {} mask : {} | masks : {}\n'.format(image_name, mask_name, task[0], np.unique(m), np.unique(mask))
        
#         print(out_str)
        sample = {'image':image , 'mask': torch.from_numpy(mask)}
        # If transform apply transformation
        if self.transform:
            sample = self.transform(sample)
        return sample
        
print('dataset.py')