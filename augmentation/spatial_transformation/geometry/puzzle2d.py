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


class Puzzle2d(torch.nn.Module):
    def __init__(self, probability: float = 1., pieces: tuple = (3, 3)):
        """
        Split elements of the input tensor in pieces. 

        Parameters
        ----------
        probability : float, optional
            Randomly adds offset to some of the elements of the input tensor with probability 
            using samples from a uniform distribution. The default is 1.0.
        pieces : tuple, optional
            Amount of pieces in rows and colums. The default is (3,3).

        Returns
        -------
        None.

        """
        super().__init__()
        self.probability = probability
        self.pieces = torch.tensor(pieces)

    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, pieces={})'.format(
            self.probability, self.pieces)

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
            buffer = inp.clone()
            rows, cols = self.pieces
            shape = torch.tensor(inp.shape)
            rest = torch.remainder(shape[-2:], self.pieces)
            rows_, cols_ = torch.div(shape[-2:], self.pieces, rounding_mode='floor')

            data = inp.unsqueeze(0)
            data = [data[..., i*rows_:(i+1)*rows_, j*cols_:(j+1)*cols_] for j in range(cols) for i in range(rows)]
            data = torch.cat(data)[torch.randperm(rows * cols)]
            inp = torch.cat([torch.cat(list(data[i*cols:(i+1)*cols, ...]), dim=-1) for i in range(rows)], dim=-2)

            if rest.sum() != 0:
                buffer[..., : shape[-2]-rest[-2], : shape[-1]-rest[-1]
                       ] = inp[..., : shape[-2]-rest[-2], : shape[-1]-rest[-1]]
                inp = buffer.clone()

        return inp
