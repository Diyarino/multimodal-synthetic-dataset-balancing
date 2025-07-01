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


class LinearTrend(torch.nn.Module):
    def __init__(self, probability: float = 1., slope: tuple = (-1., 1.), bias: tuple = (0., 0.), 
                 length: int = None, channels: int = 1):
        """
        Add trend to the elements of the input tensor.

        Parameters
        ----------
        probability : float, optional
            Apply argumenttaion with probability using samples from a uniform distribution. The default is 1.0.
        slope, bias : float or tuple, optional
            The max parameter of the linear function.
        length: int or tuple, optional
            Amount of samples.
        channels: int or tuple, optional
            Amount of randomly selected features.

        Returns
        -------
        None.

        """
        super().__init__()
        self.probability = probability
        self.slope = slope
        self.bias = bias
        self.length = length
        self.channels = channels

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
        *z, h, w = inp.unsqueeze(0).shape
        length = w if not self.length else self.length
        channels = h if not self.channels else self.channels
        
        if type(self.slope) == float:
            slope = torch.ones(h)*self.slope
        else: 
            slope = (self.slope[0] - self.slope[1]) * torch.rand(h) + self.slope[1]
            
        if type(self.bias) == float:
            bias = torch.ones(h)*self.bias
        else: 
            bias = (self.bias[0] - self.bias[1]) * torch.rand(h) + self.bias[1]
        
        if type(length) == int:
            length = torch.ones(h, dtype=torch.int32)*length
        else:
            length = torch.randint(length[0], length[1], (h,))
        
        if type(channels) == int:
            channels = torch.randperm(h)[:channels]
        else:
            channels = torch.randperm(h)[:torch.randint(channels[0], channels[1]+1, (1,))]
    
        for idx, sensor in enumerate(channels):
            start = torch.randint(0, (w-length[idx]+1).item(), (1,))
            inp[..., sensor, start:start+length[idx]] += slope[idx] * torch.arange(0, length[idx]) + bias[idx]
        return inp
    

    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, max_slope={}, max_bias={}, length={}, channels={})'.format(
            self.probability, self.slope, self.bias, self.length, self.channels)



# %% test

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    data = torch.rand(3,1000)
    augmentation = LinearTrend(probability = 1., slope = (-.05, .05), bias= (0., 0.), channels=(0,3), length = (200,500))
    plt.plot(data.permute(1,0))
    plt.plot(augmentation(data).permute(1,0))
    
    


