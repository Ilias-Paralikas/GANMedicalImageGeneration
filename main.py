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
    parser.add_argument('--epochs',type=int,default=1)
    parser.add_argument('--generator_epochs',type=int,default=1)
    parser.add_argument('--static',type=bool,default=True)

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
        path=  generator_path
    ).to(device)
    
    
    discriminator_choices ={
        'Simple': SimpleDiscriminator
    }
    discriminator_path  =   os.path.join(architecture_filepaths,'discriminator.pth')
    discriminator = discriminator_choices[hyperparameters['architecture']](
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



    gen_optimizer = torch.optim.Adam(generator.parameters(), lr= hyperparameters['gen_lr'],betas=(0.5,0.999) )
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr= hyperparameters['disc_lr'],betas=(0.5,0.999))
    loss_function = nn.BCELoss()
    
    train_GAN(generator,
              discriminator,
              gen_optimizer,
              disc_optimizer,
              loss_function,
              dataloader,
              epochs=args.epochs,
              generator_epochs=args.generator_epochs,
              device=device,
              save_frequency=1,
              versbose_frequency=args.verbose_frequency,
              static=args.static)
  
    



if __name__ =='__main__':
    main()