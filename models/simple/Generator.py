import torch
import torch.nn as nn

from models.bases.generator_base import GeneratorBase
class SimpleGenerator(GeneratorBase):
    def __init__(self,noise_shape=(128,4,4,1), channels=1024,out_channels=2,path = 'models/simple/generator.pth'):
        super(SimpleGenerator, self).__init__()
        self.path  =path
        self.noise_shape = noise_shape
        self.noise_dim  =noise_shape [0]
        self.net = nn.Sequential(
            self.create_block(self.noise_dim, channels),
            self.create_block(channels, int(channels/2)),
            self.create_block(int(channels/2), int(channels/4)),
            self.create_block(int(channels/4), int(channels/8)),
            self.create_block(int(channels/8), int(channels/16)),
            self.create_block(int(channels/16), out_channels),
            nn.Tanh()
        )
    def forward(self, x):
        return self.net(x)
        
    def create_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)            
            
        )
        
        