# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 14:10:29 2022

@author: Diyar Altinses, M.Sc.

to-do:
    - 
"""

#%% imports

import torch

# %% augmentations

class NormalizeLogScaling(torch.nn.Module):
    def __init__(self, probability: float = 1.):
        """
        Normalize the input tensor with log function.
    
        Parameters
        ----------
        probability : float, optional
            Add gain to the elements of the input tensor with probability 
            using samples from a uniform distribution. The default is 0.5.
        
        Returns
        -------
        None.
    
        """
        super().__init__()
        self.probability = probability
        
    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={})'.format(
            self.probability)
        
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Call argumentation.
    
        Parameters
        ----------
        inp : torch.Tensor
            The input array which needs some offsets.
    
        Returns
        -------
        inp :  torch.Tensor
            The modified input.
    
        """
        if self.probability>torch.rand(1):
            inp = torch.log(inp)
        return inp