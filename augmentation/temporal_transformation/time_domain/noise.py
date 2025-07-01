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


class Gausnoise(torch.nn.Module):
    def __init__(self, probability: float = 1., mean: float = 0, std: float = 1., clamp: bool = False,
                 type_: str = 'float32'):
        """
        Add gaussian noise to the elements of the input tensor.
    
        Parameters
        ----------
        probability : float, optional
            Randomly adds noise to some of the elements of the input tensor with probability 
            using samples from a uniform distribution. The default is 0.5.
        mean, p_std : float, optional
            Parameters of the gaussian noise.
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
        self.mean = mean
        self.std = std
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
        return self.__class__.__name__+'(probability={}, mean={}, std={}, clamp={})'.format(
            self.probability, self.mean, self.std, self.clamp)

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
            inp = inp + torch.randn(inp.shape, device=inp.device) * self.std + self.mean
            if self.clamp:
                inp = torch.clamp(inp, min = self.clamp[0], max = self.clamp[1])
        return inp.to(self.type)



class Gausnoise_perCH(torch.nn.Module):
    def __init__(self, probability: float = 1., mean: float = 0, std: float = 1., 
                 channels: int = 1, length: int = None, type_: str = 'float32'):
        """
        Add gaussian noise to the elements of the input tensor.

        Parameters
        ----------
        probability : float, optional
            Apply argumenttaion with probability using samples from a uniform distribution. The default is 1.0.
        mean, p_std : float or tuple, optional
            Parameters of the gaussian noise.
        length: int or tuple, optional
            Amount of samples.
        channels: int or tuple, optional
            Amount of randomly selected features.
        type_ : str, optional
            The target type of the data. torch dtype class name. The default is float32.

        Returns
        -------
        None.

        """
        super().__init__()
        self.probability = probability
        self.mean = mean
        self.std = std
        self.channels = channels
        self.length = length
        self.type = getattr(torch, type_)

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
            *z, h, w = inp.shape
            self.length = w if not self.length else self.length
            self.channels = h if not self.channels else self.channels
            
            if type(self.channels) == int:
                channels = torch.randperm(h)[:self.channels]
            else:
                channels = torch.randperm(h)[:torch.randint(self.channels[0], self.channels[1]+1, (1,))]
                
            if type(self.mean) == float:
                mean = torch.ones(h)*self.mean
            else: 
                mean = (self.mean[0] - self.mean[1]) * torch.rand(h) + self.mean[1]
                
            if type(self.std) == float:
                std = torch.ones(h)*self.std
            else: 
                std = (self.std[0] - self.std[1]) * torch.rand(h) + self.std[1]
                
            if type(self.length) == int:
                length = torch.ones(h, dtype=torch.int32)*self.length
            else:
                length = torch.randint(self.length[0], self.length[1], (h,))
            
            for idx, sensor in enumerate(channels):
                start = torch.randint(0, (w-length[idx]+1).item(), (1,))
                data = inp[..., sensor, start:start+length[idx]]
                data += torch.randn_like(data) * std[idx] + mean[idx]
        return inp.to(self.type)
    
    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, mean={}, std={}, channels = {}, length={})'.format(
            self.probability, self.mean, self.std, self.channels, self.length)



# %% test

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    data = torch.zeros(3,1000)+torch.arange(0,10, 0.01)*torch.arange(3).unsqueeze(0).permute(1,0)
    augmentation = Gausnoise_perCH(probability = 1., mean = (0, 2.5), std= (0.1, 2.6), channels=1, length= None)
    plt.plot(data.permute(1,0))
    plt.plot(augmentation(data).permute(1,0), )


