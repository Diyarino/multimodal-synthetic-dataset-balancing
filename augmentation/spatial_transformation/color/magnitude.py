# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 14:15:15 2022

@author: Diyar Altinses, M.Sc.

to-do:
    - 
"""

# %% imports

import torch

# %% augmentations


class Magnitude(torch.nn.Module):
    def __init__(self, probability: float = 1., negative: float = 0.5):
        """
        Add gain to the elements of the input tensor.

        Parameters
        ----------
        probability : float, optional
            Add gain to the elements of the input tensor with probability 
            using samples from a uniform distribution. The default is 0.5.
        negative : float, optional
            Add negation to the elements of the input tensor with probability 
            using samples from a uniform distribution. The default is 0.5.

        Returns
        -------
        None.

        """
        super().__init__()
        self.probability = probability
        self.negative = negative
        
    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, negative={})'.format(
            self.probability, self.negative)

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
        if self.probability > torch.rand(1):
            # print("square inp")
            inp = inp.square().sqrt()
            if self.negative > torch.rand(1):
                inp = -inp
        return inp
