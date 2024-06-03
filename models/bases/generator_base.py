import torch.nn as nn
import torch
from utils.visualizer import Visualizer
class GeneratorBase(nn.Module):
    def __init__(self) -> None:
        self.visualiser = Visualizer()
        super(GeneratorBase, self).__init__()
        
    def generate(self,batch=1):
        noise = torch.randn(batch,*self.noise_shape)
        return self.forward(noise)
    
    def show_generated_image(self,savefile=None):
        noise = torch.randn(1,*self.noise_shape)
        scan  = self.generate()[0].detach().numpy()
        image,mask = scan[0],scan[1]
        self.visualiser(image,mask,savefile)
