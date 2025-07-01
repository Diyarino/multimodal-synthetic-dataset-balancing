# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 14:13:40 2022

@author: Diyar Altinses, M.Sc.

to-do:
    - 
"""

# %% imports

import torch

# %% white noise

class MultiplicativeWhiteNoise(torch.nn.Module):
    def __init__(self, probability: float = 1., max_amplitude: float = 0.1, max_lines: int = 32):
        """
        Augment the image with additive white noise.
        
        Parameters
        ----------
        probability : float, optional
            Randomly adds noise to some of the elements of the input tensor with probability 
            using samples from a uniform distribution. The default is 1.0.
        amplitude : float, optional
            Amplitude of the added noise.
            
        Returns
        -------
        None.

        """
        super().__init__()
        self.probability = probability
        self.max_amplitude = max_amplitude
        self.max_lines = max_lines
        self.mode = ['singleX', 'singleY', 'multi']
        
    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, max_amplitude={}, max_lines={})'.format(
            self.probability, self.max_amplitude, self.max_lines)
    
    def singleX(self, inp: torch.tensor):
        '''
        Add periodical noise in x direction.

        Parameters
        ----------
        inp : torch.Tensor
            The input array which needs some noise.

        Returns
        -------
        inp :  torch.Tensor
            The modified input.

        '''
        max_value = inp.max()
        *rem, x_len, y_len = inp.size()
        
        lines = torch.randint(0, self.max_lines, (1,))
        amplitude = (0 - self.max_amplitude) * torch.rand(1,) + self.max_amplitude
        
        x, y = torch.linspace(0, 1, x_len), torch.linspace(0, 1, y_len)
        X, Y = torch.meshgrid(x, y)
        
        inp = (inp + amplitude*torch.cos(lines*torch.pi*X)*max_value).clamp(0,max_value).type(inp.type())
        
        return inp
    
    def singleY(self, inp: torch.tensor):
        '''
        Add periodical noise in y direction.

        Parameters
        ----------
        inp : torch.Tensor
            The input array which needs some noise.

        Returns
        -------
        inp :  torch.Tensor
            The modified input.

        '''
        max_value = inp.max()
        *rem, x_len, y_len = inp.size()
        
        lines = torch.randint(0, self.max_lines, (1,))
        amplitude = (0 - self.max_amplitude) * torch.rand(1,) + self.max_amplitude
        
        x, y = torch.linspace(0, 1, x_len), torch.linspace(0, 1, y_len)
        X, Y = torch.meshgrid(x, y)
        
        inp = (inp + amplitude*torch.cos(lines*torch.pi*Y)*max_value).clamp(0,max_value).type(inp.type())
        
        return inp
    
    def multi(self, inp: torch.tensor):
        '''
        Add periodical noise in x and y direction.

        Parameters
        ----------
        inp : torch.Tensor
            The input array which needs some noise.

        Returns
        -------
        inp :  torch.Tensor
            The modified input.

        '''
        max_value = inp.max()
        *rem, x_len, y_len = inp.size()
        
        lines = torch.randint(0, self.max_lines, (1,))
        amplitude = (0 - self.max_amplitude) * torch.rand(1,) + self.max_amplitude
        
        x, y = torch.linspace(0, 1, x_len), torch.linspace(0, 1, y_len)
        X, Y = torch.meshgrid(x, y)
        
        inp = (inp + amplitude*torch.cos(lines*torch.pi*Y)*max_value).clamp(0,max_value).type(inp.type())
        inp = (inp + amplitude*torch.sin(lines*torch.pi*(X+Y))*max_value).clamp(0,max_value).type(inp.type())
        return inp
        
        
    def forward(self, inp):
        """
        Call argumentation.

        Parameters
        ----------
        inp : torch.Tensor
            The input array which needs some noise.

        Returns
        -------
        inp :  torch.Tensor
            The modified input.

        """
        if self.probability > torch.rand(1):
            idx = torch.randint(0, 3, (1,))
            inp = getattr(self, self.mode[idx])(inp)
        return inp



# %% test

if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    from PIL import Image
    from torchvision.transforms import PILToTensor
    
    canvas = PILToTensor()(Image.open('D:/Sciebo/04_GitLab/augmentation_techniques/resources/001503.png'))
    # augmentation = MultiplicativeGaussianNoise(probability=1, mean = 50, std = 30)
    augmentation = MultiplicativeWhiteNoise(probability=1)
    new = augmentation(canvas).permute(1,2,0)
    plt.imshow(new)


