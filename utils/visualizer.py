import matplotlib.pyplot as plt
import math
import numpy as np
import torch
class Visualizer():
    def __init__(self,columns = 8,size=30, spacing=0.1):
        '''
        columns: number of columns in the grid
        size: size of the grid
        spacing: spacing between the images
        '''
        self.columns = columns
        self.size=  size
        self.spacing = spacing
        return


    def __call__(self, images=None, masks=None,savefile=None):
        
        if images is None and masks is None:
            print('please provide images, masks, or both')
            return

        # If images or masks are not a list, convert them into a list
        if not isinstance(images, list) and images is not None:
            images = [images[:,:,i].detach() for i in range(images.shape[-1])] if len(images.shape) == 3 else [images]
        if not isinstance(masks, list) and masks is not None:
            masks = masks.where(masks>0.5, torch.tensor(1.0))
            masks = masks.where(masks<0.5, torch.tensor(0.0))
            masks = [masks[:,:,i].detach()  for i in range(masks.shape[-1])] if len(masks.shape) == 3 else [masks]
        num_images = len(images) if images is not None else 0
        num_masks = len(masks) if masks is not None else 0
        total = max(num_images, num_masks)

        rows = math.ceil(total / self.columns)
        _, axes = plt.subplots(rows, self.columns, figsize=(self.size, self.size * (rows / self.columns)))
        plt.subplots_adjust(wspace=self.spacing, hspace=self.spacing)
        axes_flat = axes.flatten() if rows > 1 else [axes]

        for i in range(total):
            ax = axes_flat[i]
            if i < num_images:
                ax.imshow(images[i], cmap='gray')
            if i < num_masks:
                mask = np.ma.masked_where(masks[i] == 0, masks[i])
                ax.imshow(mask, 'prism', interpolation='none')

        for i in range(total, len(axes_flat)):
            axes_flat[i].axis('off')
        plt.tight_layout()
        if savefile is not None:
            plt.savefig(savefile)
        else:
            plt.show()
        
    def visualize_verticaly(self,dataset,index):
        index= index % dataset.slices
        images=[dataset.__getitem__(i)[0][:,:,index] for i in range(dataset.len)]
        masks=[dataset.__getitem__(i)[1][:,:,index] for i in range(dataset.len)]
        self(images,masks)
        
    
    def visualize_distribution(self,img):
        img = img.detach().cpu()
        print('mean:',torch.mean(img))
        print('std:',torch.std(img))
        
        x_flat = img.view(-1).numpy()

        plt.hist(x_flat, bins=30, edgecolor='black')
        plt.title('Histogram of Tensor Values')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.show()
                
                
      
def main():
    return

if __name__ =='__main__':
    main()