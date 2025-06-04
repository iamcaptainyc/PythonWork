import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np 
import os
from PIL import Image
import cv2


class DrawMultiMask():
    def __init__(self, image, masks, label2color=['MA':[]]):
        self.label2color=label2color
        self.image=image
        self.masks=masks
    
    def draw_mask(self, image, masks) :
        masked_image = image.copy()
        masks=np.expand_dims(masks, axis=2)
        for i,label,color in enumerate(self.label2color.items()):
            masked_image = np.where(masks==int(label),#np.repeat(masks_generated[i][:, :, np.newaxis], 3, axis=2),
                                    np.asarray(color, dtype='uint8'),
                                    masked_image)
            
            masked_image = masked_image.astype(np.uint8)
        
        return cv2.addWeighted(image, 0.3, masked_image, 0.7, 0)
    
    def plot_mask(self):
        imgs=[self.image]
        if len(self.masks.shape) == 3:
            for m in self.masks:
                imgs.append(self.draw_mask(self.image, m))
        else:
            imgs.append(self.draw_mask(self.image, self.masks))
            
            
        fig,axs = plt.subplots(1,len(imgs),sharey=True)
        for i,ax in enumerate(axs):
            ax.imshow(imgs[i])
            if i==0:
                ax.title('Origin Image')
            else:
                ax.title(f'{i}th Mask')
            ax.axis('off')
        plt.show()
        
    def set_legend(self,ax):
        patches=[mpatches.Patch(color=np.array(c)/255, label=l) for l,c in self.label2color.items()]
        #这里bbox_to_anchor表示图例的锚点，值格式为【x,y,height,width】，loc='upper left'表示图例的左上角需要与锚点重合。bbox的取值为小数，0-1表示图中0-H或0-W
        ax.legend(bbox_to_anchor=[0.82,1],loc='upper left',handles=patches)
        
    