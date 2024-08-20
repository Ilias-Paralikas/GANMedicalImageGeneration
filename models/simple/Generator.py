import torch
import torch.nn as nn

from models.bases.generator_base import GeneratorBase

class SimpleGenerator(GeneratorBase):
    def __init__(self,path,features=16,block_numbers=3):
        super(SimpleGenerator, self).__init__()
        self.noise_shape = (2,8,8,2)
        self.path = path
        channels= 2
        self.gen =  nn.Sequential(
            self._block(2,features*2**(block_numbers),4,2,1),
            *[self._block(features* 2**(block_numbers-i),features*2**(block_numbers-i-1),4,2,1) for i in range(block_numbers)],
            self._block(features,2,4,2,1),
            nn.Sigmoid()
        )
    def _block(self,in_channels,out_channels,kernel_size,stride,padding):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels,out_channels,kernel_size,stride,padding,bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.gen(x)
        x = x-0.5
        return x