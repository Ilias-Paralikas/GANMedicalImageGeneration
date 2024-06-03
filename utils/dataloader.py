import os
import numpy as np
import torch    
from .visualizer import Visualizer

    
  
    
class BreastCancerDataset(torch.utils.data.Dataset):
    def __init__(self,
                sliced_folder= 'sliced',
                dimensions=256,
                slices=64,
                image_datatype =np.float32):
        self.slices = slices
        self.dimensions = dimensions
        self.target_folder = os.path.join(sliced_folder,'sliced_'+'_'+ str(dimensions) + '_' + str(slices)+'_'+image_datatype.__name__)
        self.patient_files = [os.path.join(self.target_folder,f) for f in os.listdir(self.target_folder)]
        self.len = len(self.patient_files )
        self.visualizer= Visualizer()

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        index = index % self.len
        file = self.patient_files[index]
        data = torch.load(file)

        image_tensor = data['image']
        mask_tensor = data['mask']
        return torch.stack((image_tensor, mask_tensor))



    def show(self, index= None):
        if index is None:
            index = np.random.randint(0, self.len)
        print('Patient File: ',self.patient_files[index])
        image,mask = self.__getitem__(index)
        print(image.shape,mask.shape)
        self.visualizer(image,mask)