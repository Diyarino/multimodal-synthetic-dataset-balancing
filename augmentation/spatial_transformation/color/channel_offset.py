# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 14:12:45 2022

@author: Diyar Altinses, M.Sc.

to-do:
    - 
"""

# %% imports

import torch

# %% augmentations


class ChannelOffset(torch.nn.Module):
    def __init__(self, probability: float = 1., low: float = -1., high: float = 1., clamp: tuple = None):
        """
        Add some offset to the elements of the input tensor. Control the color of rgb image matrix (3-dim). 
    
        Parameters
        ----------
        probability : float, optional
            Randomly adds offset to some of the elements of the input tensor with probability 
            using samples from a uniform distribution. The default is 1.0.
        low, high : float, optional
            limitates the offset in a range.
        clamp : tuple, optional
            Clamp the out values between a defined range.
            
        Returns
        -------
        None.
    
        """
        super().__init__()
        self.probability = probability
        self.low = low
        self.high = high
        self.clamp = clamp
        
    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, low={}, high={}, clamp={})'.format(
            self.probability, self.low, self.high, self.clamp)
        
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Call argumentation.
    
        Parameters
        ----------
        inp : torch.Tensor
            The normalized input array which needs some color offsets.
    
        Returns
        -------
        inp :  torch.Tensor
            The modified input.
    
        """
        if self.probability>torch.rand(1):
            # print("offset")
            inp = inp + (torch.rand(inp.shape[-3], 1, 1)*(self.high-self.low)+self.low).expand(inp.size()[-3:])
            if self.clamp:
                inp = inp.clamp(*self.clamp)
        return inp