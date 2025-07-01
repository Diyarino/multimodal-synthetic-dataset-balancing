# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 14:13:40 2022

@author: Diyar Altinses, M.Sc.

to-do:
    - 
"""

# %% imports

import torch

# %% gaussian


class Gausnoise_perCH(torch.nn.Module):
    def __init__(self, probability: float = 1., mean: float = 0, p_std: float = 1., dim: int = -1,
                 type_: str = 'float32'):
        """
        Add gaussian noise per channel to the elements of the input tensor.

        Parameters
        ----------
        probability : float, optional
            Randomly adds noise to some of the elements of the input tensor with probability 
            using samples from a uniform distribution. The default is 1.0.
        mean, p_std : float, optional
            Parameters of the gaussian noise.
        dim : int, optional
            The channel which is distributed.
        type_ : str, optional
            The target type of the data. torch dtype class name. The default is float32.

        Returns
        -------
        None.

        """
        super().__init__()
        self.probability = probability
        self.mean = mean
        self.p_std = p_std
        self.dim = dim
        self.type = getattr(torch, type_)

    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, mean={}, p_std={}, dim={})'.format(
            self.probability, self.mean, self.p_std, self.dim)

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
        if self.probability > torch.rand(1):
            # print("gausinannoise per channel")
            inp_std = inp.float().std(dim=self.dim)
            inp = inp + torch.randn(inp.size()) * inp_std.unsqueeze(-1)*self.p_std + self.mean
        return inp.to(self.type)



# %% white noise



class WhiteNoise_perCH(torch.nn.Module):
    def __init__(self, probability: float = 1., amplitude: float = 1., dim: int = -1,
                 type_: str = 'float32'):
        """
        Add white noise per channel to the elements of the input tensor.

        Parameters
        ----------
        probability : float, optional
            Randomly adds noise to some of the elements of the input tensor with probability 
            using samples from a uniform distribution. The default is 1.0.
        amplitude : float, optional
            Parameters of the white noise.
        dim : int, optional
            The channel which is distributed.
        type_ : str, optional
            The target type of the data. torch dtype class name. The default is float32.

        Returns
        -------
        None.

        """
        super().__init__()
        self.probability = probability
        self.amplitude = amplitude
        self.dim = dim
        self.type = getattr(torch, type_)

    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, mean={}, p_std={}, dim={})'.format(
            self.probability, self.mean, self.p_std, self.dim)

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
        if self.probability > torch.rand(1):
            # print("gausinannoise per channel")
            inp_std = inp.float().std(dim=self.dim)
            inp = inp + torch.rand(inp.size()) * inp_std.unsqueeze(-1)*self.p_std + self.mean
        return inp.to(self.type)




