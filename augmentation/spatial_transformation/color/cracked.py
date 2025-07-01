# -*- coding: utf-8 -*-%
"""
Created on %(date)s

@author: Diyar Altinses, M.Sc.

to-do:
    - 
"""

# %% imports

import os
import torch
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import resize 
from PIL import Image

# %% application

class Cracked(torch.nn.Module):
    def __init__(self, probability: float = .5, path: str = None, type_: str = 'int32'):
        '''
        This class add some crack to the image that is looks like a broken lense.
        Btw. it removes the alpha channel.

        Parameters
        ----------
        probability : float, optional
            Randomly adds noise to some of the elements of the input tensor with probability 
            using samples from a uniform distribution. The default is 0.5.
        path : str, optional
            The path where the cracked transparent figures are stored.
        type_ : str, optional
            The target type of the data. torch dtype class name. The default is int32.

        Returns
        -------
        None.

        '''
        super().__init__()
        
        self.probability = probability
        self.path = path
        self.totensor = ToTensor()
        self.resize = resize
        self.type_ = type_
        self.type = getattr(torch, type_)
        
        path = path if path else os.path.join('..', 'resources', 'cracked')
        self.cracked = [self.totensor(Image.open(os.path.join(path, file))) for file in os.listdir(path)]
        
    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, type={})'.format(
            self.probability, self.type)
    
    def forward(self, inp: torch.Tensor):
        """
        Call argumentation.
    
        Parameters
        ----------
        inp : torch.Tensor
            The input array which needs some spiderman app.
    
        Returns
        -------
        inp :  torch.Tensor
            The modified input.
    
        """
        if torch.rand(1) <= self.probability:
            overlap = self.cracked[torch.randint(0, len(self.cracked), (1,))]
            overlap = resize(overlap, size = inp.shape[-2:])
            inp = (inp/255.)[:3] if inp.max()>1. else inp[:3]
            inp[:,:,:] = (overlap[-1] * overlap[:3]) + ((1-overlap[-1]) * inp)
            inp = (inp*255).to(self.type) if 'int' in self.type_ else inp.to(self.type)
        return inp


# %% test

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    image = torch.ones(3,256,256)
    augmentation= Cracked(probability=1, type_='float32')
    augmented = augmentation(image.clone())
    f,a = plt.subplots(1,2)
    a[0].imshow(image.permute(1,2,0))
    a[1].imshow(augmented.permute(1,2,0))






