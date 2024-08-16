from monai.transforms import LoadImage, Orientation
from skimage.transform import resize
import os
import re
import torch    
from scipy.ndimage import zoom
import numpy as np



def normalize(image):
    min_val = torch.min(image)
    max_val = torch.max(image)
    assert max_val != min_val
    image = (image - min_val) / (max_val - min_val)  # Normalize to [0, 1]
    return image

class ImagePreprocessor():
    def __init__(self,data_folder,
                    mask_folder,
                    patients_folder,
                    sliced_folder,
                    dimensions,
                    slices,
                    image_datatype,
                    mask_datatype,
                    transforms):
        self.sliced_folder=  sliced_folder

        try :
            os.makedirs(self.sliced_folder)
        except:
            print('sliced folder exists')
           
        self.data_folder = data_folder
        self.mask_folder = os.path.join(data_folder,mask_folder)
        self.patients_folder =os.path.join(data_folder,patients_folder)
        self.loader = LoadImage(image_only=True)
        self.orient = Orientation(axcodes='RAS')
        self.dimensions = dimensions
        self.slices = slices
        self.image_datatype = image_datatype
        self.mask_datatype = mask_datatype
        self.transforms  =transforms
        self.target_folder = os.path.join(self.sliced_folder,'sliced_'+'_'+ str(dimensions) + '_' + str(slices)+'_'+self.image_datatype.__name__)

      
        masks = os.listdir(self.mask_folder)
        patients = os.listdir(self.patients_folder)

        masks_numbers = self.extract_numbers(masks)



        masks_numbers = list(map(str,masks_numbers))
        masks_numbers.sort()      
          
        patients_numbers = self.extract_numbers(patients)
        patients_numbers = list(map(str,patients_numbers))
        patients_numbers.sort()
        assert (patients_numbers ==masks_numbers)
        self.numbers=  patients_numbers
        self.len = len(self.numbers)
        
    def extract_numbers(self,strings):
        numbers = []
        for s in strings:
            found_numbers = re.findall(r'\d+', s)
            numbers.extend([int(num) for num in found_numbers])
        return numbers


    def reshape(self,image,datatype):
        image = self.orient(image)
        zoom_factors = [self.dimensions/image.shape[0], self.dimensions/image.shape[1], self.slices/image.shape[2]]
        resized_image= zoom(image, zoom_factors,datatype)
        return resized_image
        
    def read_image(self, index):
        raw_image =self.read_raw_image(index)
        return self.reshape(raw_image,self.image_datatype)

    def read_mask(self, index):
        raw_mask = self.read_raw_mask(index)
        return self.reshape(raw_mask,self.mask_datatype)
    
    def read_raw_image(self, index):
        index = index % self.len
        index = self.numbers[index]
        filepath  =  os.path.join(self.patients_folder,'patient'+str(index)+'.nii.gz')
        print(filepath)
        return self.loader(filepath)

    def read_raw_mask(self, index):
        index = index % self.len
        index = self.numbers[index]
        filepath  =  os.path.join(self.mask_folder,'segmentation'+str(index)+'.nrrd')
        print(filepath)

        return self.loader(filepath)
    
    def get_raw_data(self,index):
        image = self.read_raw_image(index)
        mask = self.read_raw_mask(index)
        return image,mask
    
    def store_data(self, overwrite = False):
        try:
            os.makedirs(self.target_folder)
            overwrite = True
        except:
            print('''Probelm creating the folder, make sure that the folder does not exits\nIf it exists, set overwrite to True to overwrite''')
        if overwrite:
            for i in range(self.len):
                image = self.read_image(i)
                image_tensor = torch.from_numpy(image)
                if self.transforms:
                    image_tensor= self.transforms(image_tensor)
                mask = self.read_mask(i)                
                mask_tensor = torch.from_numpy(mask)
                
                filename = os.path.join(self.target_folder, 'patient'+str(self.numbers[i])+'.pt')
                torch.save({'image': image_tensor, 'mask': mask_tensor},filename)
                print(f'saved file {filename}')

def main():
   return

if __name__ =='__main__':
    main()