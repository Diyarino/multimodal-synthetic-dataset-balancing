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


class Replacer(torch.nn.Module):
    def __init__(self, probability: float = 1., size: tuple = (0, 0), replace_val: float = 0.):
        """
        Randomly erase elements of the input tensor in rectangular form. 

        Parameters
        ----------
        probability : float, optional
            Apply argumenttaion with probability using samples from a uniform distribution. The default is 1.0.
        size : tuple, optional
            Size of the rectangle in rows and colums. The default is (0,0).
        replace_val : float, optional
            The value of the elements in the rectangle. The default value 0. paint it black.

        Returns
        -------
        None.

        """
        super().__init__()
        self.probability = probability
        self.size = size
        self.replace_val = replace_val

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Call argumentation.

        Parameters
        ----------
        inp : torch.Tensor
            The input array which needs to be padded.

        Returns
        -------
        inp :  torch.Tensor
            The modified input.

        """
        if self.probability > torch.rand(1):
            rows, cols = inp.size()
            features = torch.randint(0, cols, (self.size[1],))
            for idx in features:
                rand_pos_y = torch.randint(0, rows-self.size[0], (1,))
                rand_len = torch.randint(0, self.size[0], (1,))
                inp[rand_pos_y: rand_pos_y + rand_len, idx] = self.replace_val
        return inp

    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, size={}, replace_val={})'.format(
            self.probability, self.size, self.replace_val)
