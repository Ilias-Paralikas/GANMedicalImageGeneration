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
    parser.add_argument('--sliced_folder',type=str,default='../sliced')
    parser.add_argument('--Reinitialize_models', type=bool, default=False, help='Float')
    parser.add_argument('--verbose_frequency', type=int, default=10, help='Float')

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

    
    generator = generator_choices[hyperparameters['architecture']](path=  generator_path).to(device)
    
    
    discriminator_choices ={
        'Simple': SimpleDiscriminator
    }
    discriminator_path  =   os.path.join(architecture_filepaths,'discriminator.pth')
    discriminator = discriminator_choices[hyperparameters['architecture']](
        in_channels=2,
        out_channels=512,
        path=discriminator_path
    ).to(device)
    
    if not args.Reinitialize_models:
        try:
            generator.load_state_dict(torch.load(generator.path,map_location=torch.device(device)))
            discriminator.load_state_dict(torch.load(discriminator.path,map_location=torch.device(device)))
            print("Weights loaded")
        except:
            print("No weights found")
        
    dataset=BreastCancerDataset(sliced_folder=args.sliced_folder)
    dataloader=  torch.utils.data.DataLoader(dataset, batch_size=hyperparameters['batch_size'], shuffle=True, num_workers=0)

    train_GAN(generator=generator,
              discriminator=discriminator,
              dataloader=dataloader,
              noise_shape=generator.noise_shape,
              epochs=hyperparameters['epochs'],
              gen_optim =getattr(torch.optim, hyperparameters['gen_optim']),
              gen_lr = hyperparameters['gen_lr'],
              disc_optim=getattr(torch.optim, hyperparameters['disc_optim']),
              disc_lr = hyperparameters['disc_lr'],
              loss_fn=getattr(nn, hyperparameters['loss_fn']),
              device= device,
              save_frequency=hyperparameters['save_frequency'],
              verbose_frequency=args.verbose_frequency)




if __name__ =='__main__':
    main()