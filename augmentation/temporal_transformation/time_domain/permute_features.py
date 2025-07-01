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


class PermuteFeatures(torch.nn.Module):
    def __init__(self, probability: float = 1.):
        """
        Permutes features of the input tensor.

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
            # print("permute channels")
            permute = torch.randperm(inp.shape[1])
            inp = inp[:, permute]
        return inp

    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={})'.format(self.probability)


class Permutation(torch.nn.Module):
    def __init__(self, probability: float = 1., pieces: int = 3, features: int = 1):
        """
        Split elements of the input tensor in pieces. 

        Parameters
        ----------
        probability : float, optional
            Apply argumenttaion with probability using samples from a uniform distribution. The default is 1.0.
        pieces : int, optional
            Amount of pieces in rows and colums. The default is 3.

        Returns
        -------
        None.

        """
        super().__init__()
        self.probability = probability
        self.pieces = pieces
        self.features = features

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
            x_range = int(inp.shape[0] / self.pieces)
            features = torch.randint(0, inp.shape[1], (self.features,))
            for feature in features:
                buffer = torch.zeros(self.pieces, x_range)
                for piece in range(self.pieces):
                    buffer[piece, :] = inp[piece*x_range:piece*x_range+x_range, feature]
                buffer = buffer[torch.randperm(buffer.shape[0])]
                inp[:, feature] = buffer.reshape(self.pieces*buffer.shape[1])
        return inp

    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, pieces={}, features={})'.format(
            self.probability, self.pieces, self.features)



class PermuteChannels(torch.nn.Module):
    def __init__(self, probability: float = 1.):
        """
        Split elements of the input tensor in pieces. 

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
            The input array which needs to be puzzled. Have to be 3 dimensional.

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
            permute = torch.randperm(inp.shape[1])
            data_permuted = inp[:, permute, :]
            inp = data_permuted if not unsqueezed else data_permuted.squeeze(0)
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
    
    
        
