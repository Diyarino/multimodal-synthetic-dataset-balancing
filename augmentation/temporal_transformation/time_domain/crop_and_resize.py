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


class CropAndResize(torch.nn.Module):
    def __init__(self, probability: float = 1.):
        """
        Apply crop and resize to the elements of the input tensor.

        Parameters
        ----------
        probability : float, optional
            Apply argumenttaion with probability using samples from a uniform distribution. The default is 1.0.


        Returns
        -------
        None.

        """
        super().__init__()
        self.probability = probability

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
            len_ = inp.size(-1)
            diff = (inp[..., 1:] + inp[..., :-1]) / 2
            dataWithInterpolation = torch.stack((inp[..., :-1], diff), dim=-1).flatten(start_dim = -2)
            startInd = torch.randint(0, int(len_ - 1), (1,))
            inp = dataWithInterpolation[..., startInd:(startInd + len_)]
        return inp

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
    
    
    
