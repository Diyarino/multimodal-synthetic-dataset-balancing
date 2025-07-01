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


class Event(torch.nn.Module):
    def __init__(self, probability: float = 1., gain: float = 1., max_sig: float = 10., features: int = 0):
        """
        Add gaussian events to the elements of the input tensor.
    
        Parameters
        ----------
        probability : float, optional
            Apply argumenttaion with probability using samples from a uniform distribution. The default is 1.0.
        gain : float, optional
            The gain of the events.
        max_sig : float, optional
            Defines the maximum value of the standard deviation.
        features: tuple or list, optional
            Amount of randomly selected features.
        
        Returns
        -------
        None.
        
        """
        super().__init__()
        self.probability = probability
        self.gain = gain
        self.features = features
        self.max_sig = max_sig
        
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Call argumentation.

        Parameters
        ----------
        inp : torch.Tensor
            The input array which needs some events.

        Returns
        -------
        inp :  torch.Tensor
            The modified input.

        """
        if self.probability>torch.rand(1):
            features = torch.randint(0, inp.shape[-1], (self.features,))
            for feature in features:
                x = torch.arange(0, inp.shape[0])
                mu, sig = torch.rand(1)*inp.shape[0], torch.rand(1)*self.max_sig
                peak = self.gain * torch.rand(1) * torch.exp(-(x - mu)**2 / (2 * (sig**2)))
                inp[:, feature] += peak
        return inp
    
    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, gain={}, max_sig={}, features={})'.format(
            self.probability, self.gain, self.max_sig, self.features)


class Step(torch.nn.Module):
    def __init__(self, probability: float = 1., gain: float = 1., features: int = 0):
        """
        Add step event to the elements of the input tensor.
    
        Parameters
        ----------
        probability : float, optional
            Apply argumenttaion with probability using samples from a uniform distribution. The default is 1.0.
        gain : float, optional
            The max gain of the events.
        features: tuple or list, optional
            Amount of randomly selected features.
            
        Returns
        -------
        None.
        
        """
        super().__init__()
        self.probability = probability
        self.gain = gain
        self.features = features
        
    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Call argumentation.

        Parameters
        ----------
        inp : torch.Tensor
            The input array which needs some step events.

        Returns
        -------
        inp :  torch.Tensor
            The modified input.

        """
        if self.probability>torch.rand(1):
            features = torch.randint(0, inp.shape[-1], (self.features,))
            for feature in features:
                start = torch.randint(0, inp.shape[0], (1,))
                step = torch.zeros(inp.shape[0])
                step[start:] += 1
                inp[:, feature] += self.gain * torch.rand(1) * step
        return inp
    
    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, gain={}, features={})'.format(
            self.probability, self.gain, self.features)

