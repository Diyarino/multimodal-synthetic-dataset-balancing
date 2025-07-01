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


class LogarithmicTrend(torch.nn.Module):
    def __init__(self, probability: float = 1., gain: float = (-1., 1.), length: int = None, channels: int = 1):
        """
        Add log event to the elements of the input tensor.

        Parameters
        ----------
        probability : float, optional
            Apply argumenttaion with probability using samples from a uniform distribution. The default is 1.0.
        gain : float or tuple, optional
            The max gain of the events. Maximum value should be 5.
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
        self.gain = gain
        self.length = length
        self.channels = channels

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Call argumentation.

        Parameters
        ----------
        inp : torch.Tensor
            The input array (2d) which needs some log events.

        Returns
        -------
        inp :  torch.Tensor
            The modified input.

        """
        *z, h, w = inp.unsqueeze(0).shape
        length = w if not self.length else self.length
        channels = h if not self.channels else self.channels
        
        if type(self.gain) == float:
            gain = torch.ones(h)*self.gain
        else: 
            gain = (self.gain[0] - self.gain[1]) * torch.rand(h) + self.gain[1]
        
        if type(length) == int:
            length = torch.ones(h, dtype=torch.int32)*length
        else:
            length = torch.randint(length[0], length[1], (h,))
        
        if type(channels) == int:
            features = torch.randperm(h)[:channels]
        else:
            features = torch.randperm(h)[:torch.randint(channels[0], channels[1]+1, (1,))]
    
        for idx, sensor in enumerate(features):
            start = torch.randint(0, (w-length[idx]+1).item(), (1,))
            inp[..., sensor, start:start+length[idx]] += gain[idx] * (torch.log(torch.arange(0, length[idx]))) 
        return inp

    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, gain={}, length={}, channels={})'.format(
            self.probability, self.gain, self.length, self.channels)


# %% test

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    data = torch.rand(3,1000)+torch.arange(0,10, 0.01)*torch.arange(3).unsqueeze(0).permute(1,0)
    augmentation = LogarithmicTrend(probability = 1., gain= (-1., 1.), length = None, channels = (0,2))
    plt.plot(augmentation(data).permute(1,0))
    
