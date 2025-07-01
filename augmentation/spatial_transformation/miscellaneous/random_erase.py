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


class RandErase(torch.nn.Module):
    def __init__(self, probability: float = 1., size: tuple = None, value: float = 0.):
        """
        Randomly erase elements of the input tensor in rectangular form. 

        Parameters
        ----------
        probability : float, optional
            Randomly adds offset to some of the elements of the input tensor with probability 
            using samples from a uniform distribution. The default is 1.0.
        size : tuple, optional
            Size of the rectangle in rows and colums. The default is None.
        erase_val : float, optional
            The value of the elements in the rectangle. The default value 0. paint it black.

        Returns
        -------
        None.

        """
        super().__init__()
        self.probability = probability
        self.size = size
        self.value = value

    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, size={}, value={})'.format(
            self.probability, self.size, self.value)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Call argumentation.

        Parameters
        ----------
        inp : torch.Tensor
            The input array which needs to be padded. Have to be 3 dimensional.

        Returns
        -------
        inp :  torch.Tensor
            The modified input.

        """
        if self.probability > torch.rand(1):
            inp = inp.clone()
            rows, cols = inp.shape[-2:]
            if self.size:
                rand_pos_y = (torch.rand(1,)*(rows-self.size[0])).int()
                rand_pos_x = (torch.rand(1,)*(cols-self.size[1])).int()
                x_vals, y_vals = (rand_pos_x, rand_pos_x + self.size[0]), (rand_pos_y, rand_pos_x + self.size[1])
            else:
                x_vals = (torch.rand(2,)*(rows)).int().sort().values
                y_vals = (torch.rand(2,)*(cols)).int().sort().values

            inp[..., y_vals[0]: y_vals[1], x_vals[0]: x_vals[1]] = self.value

        return inp
