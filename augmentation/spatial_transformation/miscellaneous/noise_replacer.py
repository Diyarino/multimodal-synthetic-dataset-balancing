# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 14:13:40 2022

@author: Diyar Altinses, M.Sc.

to-do:
    - 
"""

# %% imports

import torch

# %% augmentations


class ReplaceGausnoise(torch.nn.Module):
    def __init__(self, probability: float = 1., clamp: bool = False, type_: str = 'float32'):
        """
        Add gaussian noise to the elements of the input tensor.
    
        Parameters
        ----------
        probability : float, optional
            Randomly adds noise to some of the elements of the input tensor with probability 
            using samples from a uniform distribution. The default is 0.5.
        clamp : tuple, optional
            If data should be clamped. Default is False. If you want to clamp define as tuple with (min, max).
        type_ : str, optional
            The target type of the data. torch dtype class name. The default is float32.
                
        Returns
        -------
        None.
    
        """
        super().__init__()
        self.probability = probability
        self.clamp = clamp
        self.type = getattr(torch, type_)
        
    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, clamp={})'.format(self.probability, self.clamp)

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
        if self.probability>torch.rand(1):
            # print("gausinannoise")
            device = inp.device
            inp = torch.randn(inp.size(), device=device) * inp.max()
            if self.clamp:
                inp = torch.clamp(inp, min = self.clamp[0], max = self.clamp[1])
        return inp.to(self.type)
    
    
    
# %% white nosie


class ReplaceWhiteNoise(torch.nn.Module):
    def __init__(self, probability: float = 1., clamp: bool = False, type_: str = 'float32'):
        """
        Add white noise to the elements of the input tensor.
    
        Parameters
        ----------
        probability : float, optional
            Randomly adds noise to some of the elements of the input tensor with probability 
            using samples from a uniform distribution. The default is 0.5.
        clamp : tuple, optional
            If data should be clamped. Default is False. If you want to clamp define as tuple with (min, max).
        type_ : str, optional
            The target type of the data. torch dtype class name. The default is float32.
                
        Returns
        -------
        None.
    
        """
        super().__init__()
        self.probability = probability
        self.clamp = clamp
        self.type = getattr(torch, type_)
        
    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, clamp={})'.format(self.probability, self.clamp)

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
        if self.probability>torch.rand(1):
            # print("gausinannoise")
            inp = torch.rand(inp.size()) * inp.max()
            if self.clamp:
                inp = torch.clamp(inp, min = self.clamp[0], max = self.clamp[1])
        return inp.to(self.type)
