# -*- coding: utf-8 -*-%
"""
Created on %(date)s

@author: Diyar Altinses, M.Sc.

to-do:
    - 
"""

# %% imports

import torch

# %% application


class Dust(torch.nn.Module):
    def __init__(self, probability: float = 0.5, amount: int = 100, radius: float = 0.01):
        '''
        This class adds some water drops to the image.

        Parameters
        ----------
        probability : float, optional
            Randomly adds noise to some of the elements of the input tensor with probability 
            using samples from a uniform distribution. The default is 0.5.
        amount, radius : int, optional
            Defines the max amout and the max radius of drops in percent in relation to the image. 
            The default is 10 and 0.1.
            
        Returns
        -------
        None.

        '''
        super().__init__()
        self.probability = probability
        self.radius = radius
        self.amount = amount

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
            max_r = int((h+w)/2*self.radius)
            buffer = torch.zeros_like(inp).type(inp.type()).to(device)
            for x in range(self.amount):
                position = torch.randint(0, h, (1,)), torch.randint(0, w, (1,))
                radius = torch.randint(0, max_r, (1,))
                dist_from_center = torch.sqrt((X - position[0])**2 + (Y-position[1])**2)
                mask = (dist_from_center <= radius).expand_as(inp).to(device)
                inp = torch.where(mask == True, buffer, inp)
        return inp

# %% test
    
if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    ins = torch.ones(3,256,256)
    augment = Dust(1)
    plt.imshow(augment(ins).permute(1,2,0))
    
    
    
    
