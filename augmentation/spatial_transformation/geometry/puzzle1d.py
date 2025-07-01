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


class Puzzle1d(torch.nn.Module):
    def __init__(self, probability: float = 1., pieces: int = 3):
        """
        Split elements of the input tensor in pieces. 

        Parameters
        ----------
        probability : float, optional
            Randomly adds offset to some of the elements of the input tensor with probability 
            using samples from a uniform distribution. The default is 1.0.
        pieces : int, optional
            Amount of pieces in rows and colums. The default is 3.

        Returns
        -------
        None.

        """
        super().__init__()
        self.probability = probability
        self.pieces = pieces
        
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
            The input array which needs to be puzzled. Have to be 1 or 2 (batch) dimensional.

        Returns
        -------
        inp :  torch.Tensor
            The modified input.

        """
        if self.probability > torch.rand(1):
            rest = inp.shape[-1] % self.pieces
            inp_ = inp.unsqueeze(0) if inp.dim() == 1 else inp
            shape = inp_.shape

            if rest != 0:
                buffer = inp_[..., -rest:]
                inp_ = inp_[..., :-rest].reshape(shape[0], self.pieces, -1)
                inp_ = torch.cat((inp_[:, torch.randperm(self.pieces)].reshape(shape[0], -1), buffer), dim=1)
            else:
                inp_ = inp_.reshape(shape[0], self.pieces, -1)
                inp_ = inp_[:, torch.randperm(self.pieces)]

            inp = inp_.reshape(inp.shape)

        return inp


            