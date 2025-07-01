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


class Scale(torch.nn.Module):
    def __init__(self, probability: float = 1., lower_value: float = 0., upper_value: float = 1., features: int = 0):
        """
        Scale the elements of the input tensor.

        Parameters
        ----------
        probability : float, optional
            Apply argumenttaion with probability using samples from a uniform distribution. The default is 1.0.
        min, max : float, optional
            The min and max parameter of the linear function.
        features: tuple or list, optional
            Amount of randomly selected features.

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
            The input array which needs some trend.

        Returns
        -------
        inp :  torch.Tensor
            The modified input.

        """
        if self.probability > torch.rand(1):
            features = torch.randint(0, inp.shape[-1], (self.features,))
            for feature in features:
                inp[:, feature] = ((self.upper_value-self.lower_value)*(inp[:, feature] - inp[:,
                                   feature].min())/(inp[:, feature].max() - inp[:, feature].min()))+self.lower_value
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


class Clamp(torch.nn.Module):
    def __init__(self, probability: float = 1., min_: float = 0., max_: float = 1., features: int = 0):
        """
        Clamp the elements of the input tensor.

        Parameters
        ----------
        probability : float, optional
            Apply argumenttaion with probability using samples from a uniform distribution. The default is 1.0.
        min, max : float, optional
            The min and max parameter of the linear function.
        features: tuple or list, optional
            Amount of randomly selected features.

        Returns
        -------
        None.

        """
        super().__init__()
        self.probability = probability
        self.min = min_
        self.max = max_
        self.features = features

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Call argumentation.

        Parameters
        ----------
        inp : torch.Tensor
            The input array which needs some trend.

        Returns
        -------
        inp :  torch.Tensor
            The modified input.

        """
        if self.probability > torch.rand(1):
            features = torch.randint(0, inp.shape[-1], (self.features,))
            for feature in features:
                inp[:, feature] = inp[:, feature].clamp(self.min, self.max)
        return inp

    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, min={}, max={}, features={})'.format(
            self.probability, self.min, self.max, self.features)
