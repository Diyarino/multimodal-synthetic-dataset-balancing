# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 14:14:37 2022

@author: Diyar Altinses, M.Sc.

to-do:
    - 
"""

# %% imports

import torch

# %% # %% augmentations


class PermuteChannels(torch.nn.Module):
    def __init__(self, probability: float = 1., dim: int = 0):
        """
        Permutes channel of the input tensor.

        Parameters
        ----------
        probability : float, optional
            Randomly permutes the elements of the input tensor with probability 
            using samples from a uniform distribution. The default is 0.5.

        Returns
        -------
        None.

        """
        super().__init__()
        self.probability = probability
        self.dim = dim
        
    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, dim={})'.format(
            self.probability, self.dim)

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
            # print("permute channels")
            permute = torch.randperm(inp.shape[self.dim])
            inp = inp[permute, :]
        return inp
