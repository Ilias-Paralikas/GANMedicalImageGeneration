import torch.nn as nn
import torch
from utils.visualizer import Visualizer
class GeneratorBase(nn.Module):
    def __init__(self) -> None:
        self.visualiser = Visualizer()
        super(GeneratorBase, self).__init__()
        
    def generate(self,batch=1,device='cpu'):
        noise = torch.randn(batch,*self.noise_shape,device=device)
        return self.forward(noise)
    
    def show_generated_image(self,savefile=None,device='cpu',apply_threshold=True,noise=None):
        if noise is not None:
            scan = self.forward(noise)[0]
        else:
            scan  = self.generate(device=device)[0]
        scan = scan.cpu().detach()
        image,mask = scan[0],scan[1]
        self.visualiser(image,mask,savefile,apply_threshold)
