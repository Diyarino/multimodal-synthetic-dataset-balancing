# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 14:23:19 2022

@author: Diyar Altinses, M.Sc.

to-do:
    - 
"""

# %% imports

import torch

# %%


class FreqShift(torch.nn.Module):
    def __init__(self, probability: float = 1., shift: int = 1, features: int = 0):
        """
        Apply shift to fourier transformation to the input tensor. 

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
        self.shift = shift
        self.features = features

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
            transfer = torch.fft.fft(inp)

            features = torch.randint(0, inp.shape[-1], (self.features,))
            for feature in features:
                transfer[:, feature] = transfer[:, feature].roll(torch.randint(0, self.shift, (1,)).item())

            phase = torch.angle(transfer)
            magnitude = torch.abs(transfer)
            transfer = torch.polar(magnitude, phase)
            inp = transfer
            # inp = torch.fft.ifft(transfer)
        return inp

    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, shift={}, features={})'.format(
            self.probability, self.shift, self.features)
