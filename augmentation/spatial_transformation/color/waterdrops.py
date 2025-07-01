# -*- coding: utf-8 -*-%
"""
Created on %(date)s

@author: Diyar Altinses, M.Sc.

to-do:
    - 
"""

# %% imports

import torch
from torchvision.transforms import GaussianBlur

# %% application


class Waterdrops(torch.nn.Module):
    def __init__(self, probability: float = 0.5, amount: int = 10, radius: float = 50,
                 kernel_size: tuple = (25, 25), sigma: tuple = (15, 20)):
        '''
        This class adds some water drops to the image.

        Parameters
        ----------
        probability : float, optional
            Randomly adds noise to some of the elements of the input tensor with probability 
            using samples from a uniform distribution. The default is 0.5.
        amount, radius : int, optional
            Defines the max amout and the max radius of drops. The default is 10 and 50.
        kernel_size : int or sequence, optional
            Size of the Gaussian kernel. The default is (25 ,25).
        sigma : float or tuple of float (min, max), optional
            Standard deviation to be used for creating kernel to perform blurring. If float, sigma is fixed. 
            If it is tuple of float (min, max), sigma is chosen uniformly at random to lie in the given range.
            The default is (15, 20).

        Returns
        -------
        None.

        '''
        super().__init__()
        self.probability = probability
        self.radius = radius
        self.amount = amount
        self.blur = GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, radius={}, amount={})'.format(
            self.probability, self.radius, self.amount)

    def forward(self, inp: torch.Tensor):
        """
        Call argumentation.

        Parameters
        ----------
        inp : torch.Tensor
            The input array which needs some water drops.

        Returns
        -------
        inp :  torch.Tensor
            The modified input.

        """
        if torch.rand(1) <= self.probability:
            device = inp.device
            h, w = inp.shape[1:]
            Y, X = torch.arange(h).unsqueeze(0).permute(1, 0), torch.arange(w).unsqueeze(0)
            blurred = self.blur(inp).to(device)
            for x in range(self.amount):
                position = torch.randint(0, h, (1,)), torch.randint(0, w, (1,))
                radius = torch.randint(0, self.radius, (1,))
                dist_from_center = torch.sqrt((X - position[0])**2 + (Y-position[1])**2)
                mask = (dist_from_center <= radius).expand_as(inp).to(device)
                inp = torch.where(mask == True, blurred, inp)
        return inp
    
    
# %% test
if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    ins = torch.rand(3,256,256)
    augment = Waterdrops(1)
    plt.imshow(augment(ins).permute(1,2,0))
