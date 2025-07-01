# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 14:13:40 2022

@author: Diyar Altinses, M.Sc.

to-do:
    - 
"""

# %% imports

import torch

# %% gaussian noise

class MultiplicativeGaussianNoise(torch.nn.Module):
    def __init__(self, probability: float = 1., mean: float = 0, std: float = 1.):
        """
        Augment the image with additive gaussian noise.
        
        Parameters
        ----------
        probability : float, optional
            Randomly adds noise to some of the elements of the input tensor with probability 
            using samples from a uniform distribution. The default is 1.0.
        mean, p_std : float, optional
            Parameters of the gaussian noise.

        Returns
        -------
        None.

        """
        super().__init__()
        self.probability = probability
        self.mean = mean
        self.std = std
        
    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, mean={}, std={})'.format(self.probability, self.mean, self.std)

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
            inp = (inp + (torch.randn(inp.size()) * self.std + self.mean)).type(inp.type())
        return inp

# %% white noise

class MultiplicativeWhiteNoise(torch.nn.Module):
    def __init__(self, probability: float = 1., amplitude: float = 1):
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
        self.amplitude = amplitude
        
    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, amplitude={})'.format(self.probability, self.amplitude)

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
            inp = (inp + (torch.rand(inp.size()) * self.amplitude)).type(inp.type())
        return inp



# %% test

if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    from PIL import Image
    from torchvision.transforms import PILToTensor
    
    canvas = PILToTensor()(Image.open('D:/Sciebo/04_GitLab/augmentation_techniques/resources/001503.png'))
    augmentation = MultiplicativeGaussianNoise(probability=1, mean = 50, std = 30)
    # augmentation = MultiplicativeWhiteNoise(probability=1, amplitude = 50)
    new = augmentation(canvas).permute(1,2,0)
    plt.imshow(new)


