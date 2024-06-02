import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self,in_channels=2,out_channels=512):
        super(Discriminator, self).__init__()
        self.in_channels = in_channels  
        kernel_size = 3
        stride = 1
        padding = 1
        self.network = nn.Sequential(
            self.create_block(self.in_channels, int(out_channels / 8), kernel_size, stride, padding),
            self.create_block(int(out_channels / 8), int(out_channels / 4), kernel_size, stride, padding),
            self.create_block(int(out_channels / 4), int(out_channels / 2), kernel_size, stride, padding),
            self.create_block(int(out_channels / 2), out_channels, kernel_size, stride, padding),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(524288 , 1),
            nn.Sigmoid()
            
        )
    def forward(self,x):
        return self.network(x)
        
    def create_block(self, in_channels, out_channels, kernel_size, stride, padding):
            return nn.Sequential(
                nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.LeakyReLU(0.2, inplace=True),
                nn.AvgPool3d(kernel_size=2, stride=2)
            )
    