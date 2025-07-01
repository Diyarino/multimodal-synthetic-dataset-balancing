# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 14:11:59 2022

@author: Diyar Altinses, M.Sc.

to-do:
    - 
"""

#%% imports

import torch

# %% augmentations

class Flipping(torch.nn.Module):
    def __init__(self, probability: float = 1., dims: tuple = (0, 1)):
        """
        Flipps some of the elements of the input tensor. Be aware, it can cause errors if you flip the
        batch size or the non quadratic inputs.
    
        Parameters
        ----------
        inp : array
            
        probability : float, optional
            Randomly flipps some of the elements of the input tensor with probability 
            using samples from a uniform distribution. The default is 1.0.
        dims : tuple, optional
            The dimension where flipping is applied. The default is [0, 1].
            
        Returns
        -------
        None.
        
        """
        super().__init__()
        self.probability = probability
        self.dims = dims
        
    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, dims={})'.format(
            self.probability, self.dims)
        
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Call argumentation.

        Parameters
        ----------
        inp : torch.Tensor
            The input array which needs to be flipped.

        Returns
        -------
        inp :  torch.Tensor
            The modified input.

        """
        if self.probability>torch.rand(1):
            inp = inp.flip(self.dims)
        return inp