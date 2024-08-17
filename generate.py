from utils.dataset import BreastCancerDataset
from models.simple.Discriminator import SimpleDiscriminator
from models.simple.Generator import SimpleGenerator
from models.train import train_GAN
import torch
import torch.nn as nn
import argparse
import json
import os

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    parser = argparse.ArgumentParser(description='Script Configuration via Command Line')

    parser.add_argument('--hyperparameters_file', type=str, default='hyperparameters/hyperparameters.json', help='String')
    parser.add_argument('--save_filepaths', type=str, default='../saved_models', help='String')


    args =parser.parse_args()    
    
    os.makedirs(args.save_filepaths,exist_ok=True)

    with open(args.hyperparameters_file, 'r') as file:
            hyperparameters = json.load(file)
    architecture_filepaths =  os.path.join(args.save_filepaths,hyperparameters['architecture'])
    os.makedirs(architecture_filepaths,exist_ok=True)
    
    
    generator_choices = {
        'Simple': SimpleGenerator
    }
    generator_path  =os.path.join(architecture_filepaths,'generator.pth')

    
    generator = generator_choices[hyperparameters['architecture']](
                            path=  generator_path).to(device)
    
    

    try:
        generator.load_state_dict(torch.load(generator.path,map_location=torch.device(device)))
        print("Weights loaded")
    except:
        print("No weights found")


    generator.show_generated_image(savefile=os.path.join(architecture_filepaths,'img.jpg'),
                            device=device)


if __name__ =='__main__':
    main()