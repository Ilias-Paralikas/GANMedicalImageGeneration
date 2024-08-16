from utils import BreastCancerDataset, Visualizer
from models.train import train_GAN
from models.bases.generator_base import GeneratorBase
from models.simple.Discriminator import SimpleDiscriminator
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SLICED_FOLDER = 'sliced'
BATCHE_SIZE = 2

visualizer = Visualizer()
dataset=BreastCancerDataset(sliced_folder=SLICED_FOLDER)
dataloader=  torch.utils.data.DataLoader(dataset, batch_size=BATCHE_SIZE, shuffle=True, num_workers=0)


class OutlineGenerator(GeneratorBase):
    def __init__(self,noise_shape= (2,2,8,8)):
        super(OutlineGenerator, self).__init__()
        self.noise_shape = noise_shape
        channels= 2
        self.outline =  nn.Sequential(
            *[self.upscale_block(channels) for _ in range(5)],
            nn.Sigmoid()
        )
    def upscale_block(self,channels):
        return nn.Sequential(
            nn.ConvTranspose3d(channels,channels,4,2,1),
            nn.LeakyReLU(0.1)
        )
    def forward(self, x):
        x = self.outline(x)
        return x
    
generator = OutlineGenerator()                  
discriminator = SimpleDiscriminator()


train_GAN(generator,
              discriminator,
              dataloader,
              epochs=1,
              gen_lr = 0.0002,
              disc_lr = 0.0002,
              device= device,
              save_frequency=10,
              verbose_frequency=10)