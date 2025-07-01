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


class MovingAverage(torch.nn.Module):
    def __init__(self, probability: float = 1., window: int = 1, features: int = 0):
        """
        Calc moving average of the elements of the input tensor.

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
        if self.probability > torch.rand(1):
            features = torch.randint(0, inp.shape[-1], (self.features,))
            for feature in features:
                sums = [inp[idx: self.window + idx, feature].sum() for idx in range(inp.shape[0] - self.window)]
                inp[self.window:, feature] = torch.tensor(sums)/self.window
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




class Smoothing(torch.nn.Module):
    def __init__(self, probability: float = 1., lower_limit: float = 0.):
        """
        Calc dynamic time warping of the elements of the input tensor.
    
        Parameters
        ----------
        probability : float, optional
            Apply argumenttaion with probability using samples from a uniform distribution. The default is 1.0.
        lower_limit : float, optional
            The window warping val.
            
        Returns
        -------
        None.
        
        """
        super().__init__()
        self.probability = probability
        self.lower_limit = lower_limit
        
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
            if inp.ndim == 2:
                inp = inp.unsqueeze(0)
                unsqueezed = True
            else:
                unsqueezed = False

            # factor = torch.rand(1, device=inp.device)
            # factor = 0.4
            factor = torch.FloatTensor(1).uniform_(self.lower_limit, 1)

            conv2 = torch.nn.Conv1d(inp.size(1), inp.size(1), 3, padding=1, groups=inp.size(1),
                                    padding_mode='replicate', bias=False).to(inp.device)

            kernel = torch.tensor((factor / 2, 1 - factor, factor / 2), device=inp.device).expand(inp.size(1), 1, -1)
            conv2.weight.inp = kernel

            with torch.no_grad():
                ret_data = conv2(inp)
                ret_data[:, :, 0] = inp[:, :, 0]
                ret_data[:, :, -1] = inp[:, :, -1]

            inp = ret_data if not unsqueezed else ret_data.squeeze(0)
        return inp
    
    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, lower_limit={})'.format(
            self.probability, self.lower_limit)
    