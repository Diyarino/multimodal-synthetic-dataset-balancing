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


class Flipping(torch.nn.Module):
    def __init__(self, probability: float = 1., dims: tuple = (1, 0)):
        """
        Flipps some of the elements of the input tensor.

        Parameters
        ----------
        probability : float, optional
            Apply argumenttaion with probability using samples from a uniform distribution. The default is 1.0.
        dims : tuple, optional
            The dimension where flipping is applied. The default is (0, 1).

        Returns
        -------
        None.

        """
        super().__init__()
        self.probability = probability
        self.dims = dims

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Call argumentation.

        Parameters
        ----------
        inp : torch.Tensor
            The input array which needs to be flipped.

        Returns
        -------
        inp :  torch.Tensor
            The modified input.

        """
        if self.probability > torch.rand(1):
            inp = inp.flip(self.dims)
        return inp

    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, dims={})'.format(
            self.probability, self.dims)
    
    
class LeftToRightFlipping(torch.nn.Module):
    def __init__(self, probability: float = 1.,):
        """
        Randomly apply left ti right flipping to time series data.

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
            The input array which needs to be padded.

        Returns
        -------
        inp :  torch.Tensor
            The modified input.

        """
        if self.probability > torch.rand(1):
            if inp.ndim == 2:
                inp = inp.unsqueeze(0)
                unsqueezed = True
            else:
                unsqueezed = False
            indices = range(inp.size()[2] - 1, -1, -1)
            dataFlipped = inp[:, :, indices]
            inp = dataFlipped if not unsqueezed else dataFlipped.squeeze(0)
        return inp
    
    
class BiDirectionalFlipping(torch.nn.Module):
    def __init__(self, probability: float = 1.,):
        """
        Randomly apply bidirectional flipping to time series data.

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
            The input array which needs to be padded.

        Returns
        -------
        inp :  torch.Tensor
            The modified input.

        """
        if self.probability > torch.rand(1):
            indices = range(inp.size()[-1] - 1, -1, -1)
            inp = inp[..., indices]
        return inp
    
    
class BiDirectionalFlipping_channelwise(torch.nn.Module):
    def __init__(self, probability: float = 1.,):
        """
        Randomly apply bidirectional flipping channelwise to time series data.

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
            The input array which needs to be padded.

        Returns
        -------
        inp :  torch.Tensor
            The modified input.

        """
        if self.probability > torch.rand(1):
            inp = inp * -1
        return inp
