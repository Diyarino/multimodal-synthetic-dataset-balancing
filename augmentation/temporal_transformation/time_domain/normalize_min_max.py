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

class NormalizeMinMax(torch.nn.Module):
    def __init__(self, probability: float = 1., min_val = 0., max_val = 1.):
        """
        Normalize the input tensor between min and max values.
    
        Parameters
        ----------
        probability : float, optional
            Add gain to the elements of the input tensor with probability 
            using samples from a uniform distribution. The default is 0.5.
        min_val, max_val: float, optional
            The range of the normalized data.
        
        Returns
        -------
        None.
    
        """
        super().__init__()
        self.probability = probability
        self.min_value = min_val
        self.max_value = max_val
        
    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, min={}, max={})'.format(
            self.probability, self.min_value, self.max_value)
        
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Call argumentation.
    
        Parameters
        ----------
        inp : torch.Tensor
            The input array which needs some normalization.
    
        Returns
        -------
        inp :  torch.Tensor
            The modified input.
    
        """
        if self.probability>torch.rand(1):
            inp = (inp-inp.min())/(inp.max()-inp.min())*(self.max_value-self.min_value)+self.min_value
        return inp