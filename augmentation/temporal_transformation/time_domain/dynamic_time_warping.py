# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 14:58:15 2022

@author: Diyar Altinses, M.Sc.

to-do:
    - 
"""

#%% imports

import math
import torch

#%% argumentations

class DynamicTimeWarping(torch.nn.Module):
    def __init__(self, probability: float = 1., window: int = 1, features: int = 0):
        """
        Calc dynamic time warping of the elements of the input tensor.
    
        Parameters
        ----------
        probability : float, optional
            Apply argumenttaion with probability using samples from a uniform distribution. The default is 1.0.
        window : int, optional
            The window length to calc the average.
        features: tuple or list, optional
            Amount of randomly selected features.
            
        Returns
        -------
        None.
        
        """
        super().__init__()
        self.probability = probability
        self.window = window
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
        if self.probability>torch.rand(1):
            features = torch.randint(0, inp.shape[-1], (self.features,))
            for feature in features:
                inp = inp
        return inp
    
    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, window={}, features={})'.format(
            self.probability, self.window, self.features)
    
    
class WindowWarping(torch.nn.Module):
    def __init__(self, probability: float = 1., warping_factor: float = 0.5):
        """
        Calc dynamic time warping of the elements of the input tensor.
    
        Parameters
        ----------
        probability : float, optional
            Apply argumenttaion with probability using samples from a uniform distribution. The default is 1.0.
        warping_factor : float, optional
            The window warping val.
            
        Returns
        -------
        None.
        
        """
        super().__init__()
        self.probability = probability
        self.warping_factor = warping_factor
        
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
        if self.probability>torch.rand(1):

            length = inp.size(-1)
            warping_no = int(length * self.warping_factor)
            remainer_no = length - warping_no

            # make sure warping_no is equal number
            if warping_no % 2 == 1:
                warping_no -= 1
                remainer_no += 1

            pattern = torch.cat((
                torch.zeros(remainer_no),
                torch.tensor([1, -1]).repeat(int(warping_no / 2)))
            )
            pattern = pattern[torch.randperm(length)]

            items = []
            differences = (inp[..., 1:] + inp[..., :-1]) / 2
            for i, item in enumerate(pattern):
                if item == 1:
                    diff = differences[..., i] if not i == (length - 1) else inp[..., i]
                    
                    items.append(inp[..., i])
                    items.append(diff)
                elif item == -1:
                    pass
                else:
                    items.append(inp[..., i])

            inp = torch.stack(items, dim=-1).to(inp.device) 
            
        return inp
    
    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, warping_factor={})'.format(
            self.probability, self.warping_factor)
    


class MagnitudeWarping(torch.nn.Module):
    def __init__(self, probability: float = 1., sigma: float = 0.1):
        """
        Calc dynamic time warping of the elements of the input tensor.
    
        Parameters
        ----------
        probability : float, optional
            Apply argumenttaion with probability using samples from a uniform distribution. The default is 1.0.
        sigma : float, optional
            The window warping val.
            
        Returns
        -------
        None.
        
        """
        super().__init__()
        self.probability = probability
        self.sigma = sigma
        
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
        if self.probability>torch.rand(1):

            length = inp.size(-1)
            channels = inp.size(-2)

            x_vals = torch.linspace(0, 2 * math.pi * 2, length).repeat(channels, 1)
            # add phase distortion
            x_vals += 2 * math.pi * torch.rand(length)
            
            mid = length //2
            
            test1 = torch.sin(x_vals[..., :mid])
            test2 = torch.sin(x_vals[...,mid:])

            sin_vals = torch.cat((test1, test2), dim = -1).to(inp.device)

            inp = inp * (1 + self.sigma * sin_vals)

        return inp
    
    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, sigma={})'.format(
            self.probability, self.sigma)


    
    