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


class Seasonality(torch.nn.Module):
    def __init__(self, probability: float = 1., gain: float = 1., frequency: float = 1, length: int = None, 
                 channels: int = 0):
        """
        Add seasonality to the elements of the input tensor.

        Parameters
        ----------
        probability : float, optional
            Apply argumenttaion with probability using samples from a uniform distribution. The default is 1.0.
        gain : float or tuple, optional
            The max gain of the events.
        frequency : float or tuple, optional
            Defines the maximum value of the frequency.
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
        self.frequency = frequency
        self.channels = channels
        self.length = length
        self.pi = torch.pi

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
        *z, h, w = inp.shape
        self.length = w if not self.length else self.length
        self.channels = h if not self.channels else self.channels
        
        if type(self.gain) == float:
            gain = torch.ones(h)*self.gain
        else: 
            gain = (self.gain[0] - self.gain[1]) * torch.rand(h) + self.gain[1]
            
        if type(self.frequency) == float:
            frequency = torch.ones(h)*self.frequency
        else: 
            frequency = (self.frequency[0] - self.frequency[1]) * torch.rand(h) + self.frequency[1]
        
        if type(self.length) == int:
            length = torch.ones(h, dtype=torch.int32)*self.length
        else:
            length = torch.randint(self.length[0], self.length[1], (h,))
        
        if type(self.channels) == int:
            channels = torch.randperm(h)[:self.channels]
        else:
            channels = torch.randperm(h)[:torch.randint(self.channels[0], self.channels[1]+1, (1,))]
    
        for idx, sensor in enumerate(channels):
            start = torch.randint(0, (w-length[idx]+1).item(), (1,))
            sine = gain[idx] * torch.sin(torch.linspace(0, 1, length[idx]) * 2 * self.pi * frequency[idx])
            inp[..., sensor, start:start+length[idx]] += sine
        return inp
        

    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, gain={}, frequency={}, channels={}, length={})'.format(
            self.probability, self.gain, self.frequency, self.channels, self.length)



# %% test

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    data = torch.rand(3,1000)+torch.arange(0,10, 0.01)*torch.arange(3).unsqueeze(0).permute(1,0)
    augmentation = Seasonality(probability = 1., gain = (2, 2.5), frequency= (0.1, 5.6), channels=1, length= None)
    plt.plot(data.permute(1,0))
    plt.plot(augmentation(data).permute(1,0))

