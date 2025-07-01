# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 14:17:08 2022

@author: Diyar Altinses, M.Sc.

to-do:
    - 
"""

# %% imports

import torch

# %% augmentations


class RandomReplacer(torch.nn.Module):
    def __init__(self, probability: float = 1., values: list = [0., 1.], type_: str = 'float32'):
        """
        Add some offset to the elements of the input tensor. Control brightness of rgb image matrix. 

        Parameters
        ----------
        probability : float, optional
            Randomly adds offset to some of the elements of the input tensor with probability 
            using samples from a uniform distribution. The default is 1.0.
        low, high : float, optional
            limitates the offset in a range.
        type_ : str, optional
            The target type of the data. torch dtype class name. The default is float32.

        Returns
        -------
        None.

        """
        super().__init__()
        self.probability = probability
        self.values = values
        self.type = getattr(torch, type_)

    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, value={})'.format(
            self.probability, self.values)

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
            # print("offset")
            idx = torch.randint(low=0, high=len(self.values), size=(1,)).item()
            inp = inp*0+self.values[idx]
        return inp.to(self.type)
