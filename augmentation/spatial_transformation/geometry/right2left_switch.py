# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 14:16:21 2022

@author: Diyar Altinses, M.Sc.

to-do:
    - 
"""

# %% imports

import torch

# %% augmentations


class Switch(torch.nn.Module):
    def __init__(self, probability: float = 1., amount: float = 0.2, dim: int = 1):
        """
        Switch elements of the input tensor from right to left or vice versa.

        Parameters
        ----------
        probability : float, optional
            Randomly adds offset to some of the elements of the input tensor with probability 
            using samples from a uniform distribution. The default is 1.0.
        amount : float, optional
            Amount of image to switch from right to left or vice versa. The default is 0.2.

        Returns
        -------
        None.

        """
        super().__init__()
        self.probability = probability
        self.amount = amount
        self.dim = dim

    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, amount={}, dim={})'.format(
            self.probability, self.amount, self.dim)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Call argumentation.

        Parameters
        ----------
        inp : torch.Tensor
            The input array which needs to be puzzled. Have to be 3 dimensional.

        Returns
        -------
        inp :  torch.Tensor
            The modified input.

        """
        if self.probability > torch.rand(1):
            width = inp.shape[self.dim]
            amount = int(width * self.amount)
            if 0.5 > torch.rand(1):
                inp = torch.cat([inp.narrow(self.dim, amount, width-amount),
                                inp.narrow(self.dim, 0, amount)], dim=self.dim)
            else:
                inp = torch.cat([inp.narrow(self.dim, width-amount, amount),
                                inp.narrow(self.dim, 0, width-amount)], dim=self.dim)
        return inp
