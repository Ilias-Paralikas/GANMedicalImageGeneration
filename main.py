from utils.dataloader import BreastCancerDataset
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
    parser.add_argument('--save_filepaths', type=str, default='saved_models', help='String')


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

    
    gen = generator_choices[hyperparameters['architecture']](
                            noise_shape=hyperparameters['noise_shape'],
                            path=  generator_path).to(device)
    
    
    discriminator_choices ={
        'Simple': SimpleDiscriminator
    }
    discriminator_path  =   os.path.join(architecture_filepaths,'discriminator.pth')
    disc = discriminator_choices[hyperparameters['architecture']](
        in_channels=2,
        out_channels=512,
        path=discriminator_path
    ).to(device)
    
    try:
        gen.load_state_dict(torch.load(gen.path))
        disc.load_state_dict(torch.load(disc.path))
    except:
        print("No weights found")
        
    dataset=BreastCancerDataset(dimensions=256,slices=64)
    dataloader=  torch.utils.data.DataLoader(dataset, batch_size=hyperparameters['batch_size'], shuffle=True, num_workers=0)
    # train_GAN(generator=gen,
    #           discriminator=disc,
    #           dataloader=dataloader,
    #           noise_shape=hyperparameters['noise_shape'],
    #           epochs=hyperparameters['epochs'],
    #           gen_optim =getattr(torch.optim, hyperparameters['gen_optim']),
    #           gen_lr = hyperparameters['gen_lr'],
    #           disc_optim=getattr(torch.optim, hyperparameters['disc_optim']),
    #           disc_lr = hyperparameters['disc_lr'],
    #           loss_fn=getattr(nn, hyperparameters['loss_fn']),
    #           device= device)
    gen.show_generated_image(savefile=os.path.join(architecture_filepaths,'img.jpg'))



if __name__ =='__main__':
    main()