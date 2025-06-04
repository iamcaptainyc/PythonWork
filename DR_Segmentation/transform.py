import albumentations.augmentations.geometric as G
import albumentations.augmentations.transforms as A
import albumentations.augmentations.crops as C
import torchvision.transforms as T
import torch
import numpy as np
import random


class OneOf:
    def __init__(self, transforms, choose_one_of=None):
        self.transforms = transforms
        self.choose_one_of = choose_one_of
        transforms_ps = [t.p for t in self.transforms]
        s = sum(transforms_ps)
        self.transforms_ps = [t / s for t in transforms_ps]

    def __call__(self, sample):
        if self.choose_one_of:
            t = self.transforms[self.choose_one_of]
        else:
            t = random.choices(self.transforms, weights=self.transforms_ps)[0]
        return t(sample)

class ToNumpy(object):
    
    def __call__(self, sample):
        image, masks = sample['image'], sample['masks']
        image=image.numpy()
        masks=[m.numpy() for m in masks]
        image=np.transpose(image,(1,2,0))
        new_sample = {'image': image, 'masks': masks}
        return new_sample
    
class Resize(object):
    def __init__(self, size, p=1):
        self.size=size
        self.p=p
    
    def __call__(self, sample):
        if self.size:
            image, masks = sample['image'], sample['masks']
            augmented=G.Resize(height=self.size[0], width=self.size[1])(image=image,masks=masks)
            new_sample = {'image': augmented['image'], 'masks': augmented['masks']}
            return new_sample
        return sample
    
class RandomCrop(object):
    def __init__(self, size, p=1):
        self.size=size
        self.p=p
    
    def __call__(self, sample):
        if self.size:
            image, masks = sample['image'], sample['masks']
            augmented=C.RandomCrop(height=self.size[0], width=self.size[1])(image=image,masks=masks)
            new_sample = {'image': augmented['image'], 'masks': augmented['masks']}
            return new_sample
        return sample

class RandomRotate90(object):
    '''
        Randomly rotates an image
    '''
    def __call__(self, sample):
        image, masks = sample['image'], sample['masks']
        augmented=G.RandomRotate90()(image=image,masks=masks)
        new_sample = {'image': augmented['image'], 'masks': augmented['masks']}
        return new_sample


class HorizontalFlip(object):

    def __call__(self, sample):
        image, masks = sample['image'], sample['masks']
        augmented=G.HorizontalFlip()(image=image,masks=masks)
        new_sample = {'image': augmented['image'], 'masks': augmented['masks']}
        return new_sample

class VerticalFlip(object):

    def __call__(self, sample):
        image, masks = sample['image'], sample['masks']
        augmented=G.VerticalFlip()(image=image,masks=masks)
        new_sample = {'image': augmented['image'], 'masks': augmented['masks']}
        return new_sample
    
class ColorJitter(object):
    
    def __call__(self, sample):
        image, masks = sample['image'], sample['masks']
        augmented=A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0)(image=image)
        new_sample = {'image': augmented['image'], 'masks': masks}
        return new_sample

class CropTo4(object):
    def __init__(self, size):
        self.size=size

    def __call__(self, sample):
        if self.size:
            image, masks = sample['image'], sample['masks']
            augmented_image=self.crop_to_4(image)
            augmented_masks=[]
            for mask in masks:
                augmented_masks.append(self.crop_to_4(mask))
            new_sample = {'image': augmented_image, 'masks': augmented_masks}
            return new_sample
        else:
            return sample
        
    def crop_to_4(self, img):
        y_mid = img.shape[0]//2
        y_end = img.shape[0]
        x_mid = img.shape[1]//2
        x_end = img.shape[1]
        img_lt = img[0:y_mid, 0:x_mid]
        img_lb = img[y_mid:y_end, 0:x_mid]
        img_rt = img[0:y_mid, x_mid:x_end]
        img_rb = img[y_mid:y_end, x_mid:x_end]
        
        img_cat = np.stack([img_lt, img_rt, img_lb, img_rb], axis=0)
        return img_cat
        
class Grid(object):
    def __init__(self, grid_size):
        self.grid_size=grid_size
        
    def __call__(self, sample):
        if self.grid_size:
            image, masks = sample['image'], sample['masks']
            augmented_image=self.make_pieces(image)
            augmented_masks=[]
            for mask in masks:
                augmented_masks.append(self.make_pieces(mask))
            new_sample = {'image': augmented_image, 'masks': augmented_masks}
            return new_sample
        else:
            return sample
        
    def make_pieces(self, img):
        y_coord = np.arange(0, img.shape[0], self.grid_size)
        x_coord = np.arange(0, img.shape[1], self.grid_size)
    
        img_pieces=[]
        for y in y_coord:
            row=[]
            for x in x_coord:
                row.append(img[y:y+self.grid_size, x:x+self.grid_size])
            img_pieces.append(np.stack(row, axis=0))
        
        img_cat = np.stack(img_pieces, axis=0)
        return img_cat
        
    
class ToTensor(object):
    
    def __call__(self, sample):
        image, masks = sample['image'], sample['masks']
        if len(image.shape) == 4:
            image=np.transpose(image, (0,3,1,2))
        else:
            image=np.transpose(image,(2,0,1))
        image=torch.from_numpy(image)
        masks=[torch.from_numpy(np.ascontiguousarray(m)) for m in masks]
        new_sample = {'image': image, 'masks': masks}
        return new_sample
    
class Normalize(object):
    def __init__(self):
        self.channel_stats=dict(mean = [0.425753653049469, 0.29737451672554016, 0.21293757855892181],
                         std = [0.27670302987098694, 0.20240527391433716, 0.1686241775751114])
    
    def __call__(self, sample):
        image, masks = sample['image'], sample['masks']
        augmented=T.Normalize(**self.channel_stats)(image)
        new_sample = {'image': augmented, 'masks': masks}
        return new_sample

print('transforms.py')