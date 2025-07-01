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


class MagnitudeScaling(torch.nn.Module):
    def __init__(self, probability: float = 1., lower_value: float = 0., upper_value: float = 1., features: int = 0):
        """
        Randomly scale magnitude. 

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
        self.lower_value = lower_value
        self.upper_value = upper_value
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
            phase = torch.angle(transfer)
            magnitude = torch.abs(transfer)

            features = torch.randint(0, inp.shape[-1], (self.features,))
            for feature in features:
                magnitude[:, feature] = ((self.upper_value-self.lower_value) *
                                         (magnitude[:, feature] - magnitude[:, feature].min()) /
                                         (magnitude[:, feature].max() - magnitude[:, feature].min())
                                         )+self.lower_value

            transfer = torch.polar(magnitude, phase)
            inp = torch.fft.ifft(transfer)
        return inp

    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, lower_value={}, upper_value={}, features={})'.format(
            self.probability, self.lower_value, self.upper_value, self.features)
