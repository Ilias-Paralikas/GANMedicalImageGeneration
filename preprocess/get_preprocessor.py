from .preprocessor import ImagePreprocessor, normalize
import json
import numpy as np
import os
def get_preprocessor(config_filepath = './preprocess/config.json',
                     parent_filepath='./'):
    
    with open(config_filepath, 'r') as json_file:
        config = json.load(json_file)

    preprocessor = ImagePreprocessor(
        data_folder=os.path.join(parent_filepath,config['data_folder']),
        mask_folder=config['mask_folder'],
        patients_folder=config['patients_folder'],
        sliced_folder=config['sliced_folder'],
        dimensions=config['dimensions'],
        slices=config['slices'],
        image_datatype=eval(config['image_datatype']),
        mask_datatype=eval(config['mask_datatype']),
        transforms=normalize
    )
    
    return preprocessor