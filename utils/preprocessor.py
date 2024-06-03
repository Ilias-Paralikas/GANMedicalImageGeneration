from monai.transforms import LoadImage, Orientation, Resize
from skimage.transform import resize
import os
import re
import torch    
from scipy.ndimage import zoom
import argparse


def normalize(image):
    image = image - torch.min(image)
    image = image / torch.max(image)
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
                    transforms= normalize,):
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
        self.sliced_folder=  sliced_folder

      
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


    def preprocess(self,image,datatype):
        image = self.orient(image)
        zoom_factors = [self.dimensions/image.shape[0], self.dimensions/image.shape[1], self.slices/image.shape[2]]
        resized_image= zoom(image, zoom_factors,datatype)
        return resized_image
        
    def read_image(self, index):
        raw_image =self.read_raw_image(index)
        return self.preprocess(raw_image,self.image_datatype)

    def read_mask(self, index):
        raw_mask = self.read_raw_mask(index)
        return self.preprocess(raw_mask,self.mask_datatype)
    
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
                print(f'saved file {filename}')
                torch.save({'image': image_tensor, 'mask': mask_tensor},filename)
     
  
def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_folder', type=str, default='data', help='String')
    parser.add_argument('--mask_folder', type=str, default='masks', help='String')
    parser.add_argument('--patients_folder', type=str, default='patients', help='String')
    parser.add_argument('--sliced_folder', type=str, default='sliced', help='String')
    parser.add_argument('--dimensions', type=int, default=256, help='Int')
    parser.add_argument('--slices', type=int, default=64, help='Int')
    parser.add_argument('--overwrite', type=bool, default=False, help='Bool')
    parser.add_argument('--image_datatype', type=str, default='np.float32', help='String')
    parser.add_argument('--mask_datatype', type=str, default='np.bool_', help='String')
    
    args = parser.parse_args()
    
    preprocessor = ImagePreprocessor(data_folder=args.data_folder,
                                        mask_folder=args.mask_folder,
                                        patients_folder=args.patients_folder,
                                        sliced_folder=args.sliced_folder,
                                        dimensions=args.dimensions,
                                        slices=args.slices,
                                        image_datatype=eval(args.image_datatype),
                                        mask_datatype=eval(args.mask_datatype))
    preprocessor.store_data(args.overwrite)
    
    
if __name__ =='__main__':
    main()