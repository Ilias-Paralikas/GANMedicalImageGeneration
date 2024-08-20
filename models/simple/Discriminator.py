import torch
import torch.nn as nn
class SimpleDiscriminator(nn.Module):
    def __init__(self,path,features=16,block_numbers=4):
        super(SimpleDiscriminator,self).__init__()
        self.path= path
        self.disc =  nn.Sequential(
            nn.Conv3d(2,features,4,2,1),
            nn.LeakyReLU(0.2),
            *[self._block(features* 2**i,features*2**(i+1),4,2,1) for i in range(block_numbers)],
            nn.Conv3d(features*2**(block_numbers),1,(10,10,4),2,1),
            nn.Sigmoid()

        )
    def _block(self,in_channels,out_channels,kernel_size,stride,padding):
        return nn.Sequential(
            nn.Conv3d(in_channels,out_channels,kernel_size,stride,padding,bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self,x):
        return self.disc(x).view(-1)
                      
        