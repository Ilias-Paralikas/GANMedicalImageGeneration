from monai.transforms import LoadImage, Orientation, Resize
import os
import numpy as np
import re
import torch    
from scipy.ndimage import zoom
from visualizer import Visualizer

class ImagePreprocessor():
    def __init__(self,data_folder,
                    mask_folder,
                    patients_folder,
                    dimensions,
                    slices,
                    image_datatype,
                    mask_datatype,
                    mode):
        self.data_folder = data_folder
        self.mask_folder = os.path.join(data_folder,mask_folder)
        self.patients_folder =os.path.join(data_folder,patients_folder)
        self.loader = LoadImage(image_only=True)
        self.orient = Orientation(axcodes='RAS')
        self.dimensions = dimensions
        self.slices = slices
        self.image_datatype = image_datatype
        self.mask_datatype = mask_datatype
        self.target_folder = os.path.join(data_folder,'sliced_' +mode+'_'+ str(dimensions) + '_' + str(slices)+'_'+self.image_datatype.__name__)
        self.mode = mode

        masks = os.listdir(self.mask_folder)
        patients = os.listdir(self.patients_folder)

        masks_numbers = self.extract_numbers(masks)
        masks_numbers = np.sort(masks_numbers)
        patients_numbers = self.extract_numbers(patients)
        patients_numbers = np.sort(patients_numbers)
        assert (patients_numbers ==masks_numbers).all()
        self.numbers=  patients_numbers
        self.len = len(self.numbers)
        
    def extract_numbers(self,strings):
        numbers = []
        for s in strings:
            found_numbers = re.findall(r'\d+', s)
            numbers.extend([int(num) for num in found_numbers])
        return numbers


    def preprocess(self,filepath,datatype):
        image = self.loader(filepath)
        image = self.orient(image)
        zoom_factors = [self.dimensions/image.shape[0], self.dimensions/image.shape[1], self.slices/image.shape[2]]
        resized_image= zoom(image, zoom_factors,datatype)
        return resized_image
        
    def read_raw_image(self, index):
        index = index % self.len
        index = self.numbers[index]
        filepath  =  os.path.join(self.patients_folder,'patient'+str(index)+'.nii.gz')
        return self.preprocess(filepath,self.image_datatype)

    def read_raw_mask(self, index):
        index = index % self.len
        index = self.numbers[index]
        filepath  =  os.path.join(self.mask_folder,'segmentation'+str(index)+'.nrrd')
        return self.preprocess(filepath,self.mask_datatype)

    
    def store_data(self, overwrite = False):
        try:
            os.makedirs(self.target_folder)
            overwrite = True
        except:
            print('''Probelm creating the folder, make sure that the folder does not exits\nIf it exists, set overwrite to True to overwrite''')
        if overwrite:
            for i in range(self.len):
                image = self.read_raw_image(i)
                image_tensor = torch.from_numpy(image)
                mask = self.read_raw_mask(i)                
                mask_tensor = torch.from_numpy(mask)
                
                filename = os.path.join(self.target_folder, 'patient'+str(self.numbers[i])+'.pt')
                print(f'saved file {filename}')
                torch.save({'image': image_tensor, 'mask': mask_tensor},filename)
      
            
    def get_stats(self):
        return self.len,self.numbers,self.target_folder
    
    
class Dataloader:
    def __init__(self,
                 data_folder = 'data',
                mask_folder ='masks',
                patients_folder = 'patients',
                dimensions=256,
                slices=64,
                overwrite=False,
                image_datatype =np.int16,
                mask_datatype = np.bool_,
                mode= 'constant'):
        self.slices = slices
        self.dimensions = dimensions
        preprocessor = ImagePreprocessor(data_folder, mask_folder, patients_folder, dimensions, slices, image_datatype, mask_datatype,mode)
        preprocessor.store_data(overwrite)
        
        self.len,self.numbers,self.datafolder = preprocessor.get_stats()
        self.patient_files = [os.path.join(self.datafolder,f) for f in os.listdir(self.datafolder)]
    

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        index = index % self.len
        file = self.patient_files[index]
        data = torch.load(file)

        image_tensor = data['image']
        mask_tensor = data['mask']
        return image_tensor, mask_tensor



# loader=Dataloader(patients_folder='101',mask_folder='masks_101',dimensions=512,slices=152,overwrite=True)
