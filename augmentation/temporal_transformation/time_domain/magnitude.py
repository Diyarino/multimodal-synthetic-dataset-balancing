# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 14:58:15 2022

@author: Diyar Altinses, M.Sc.

to-do:
    - 
"""

# %% imports

import torch

# %% argumentations


class Magnitude(torch.nn.Module):
    def __init__(self, probability: float = 1., negative: float = 0.5):
        """
        Add gain to the elements of the input tensor.

        Parameters
        ----------
        probability : float, optional
            Apply argumenttaion with probability using samples from a uniform distribution. The default is 1.0.
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
            # print("absolut inp")
            inp = inp.abs()
            if self.negative > torch.rand(1):
                inp = -inp
        return inp

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
