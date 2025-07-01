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
            inp = inp + torch.randn(inp.size()) * self.std + self.mean
            if self.clamp:
                inp = torch.clamp(inp, min = self.clamp[0], max = self.clamp[1])
        return inp.to(self.type)

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
            inp = torch.rand(inp.size()) * inp.max()
            if self.clamp:
                inp = torch.clamp(inp, min = self.clamp[0], max = self.clamp[1])
        return inp.to(self.type)


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
        if self.probability>torch.rand(1):
            # print("gausinannoise per channel")
            inp_std=inp.float().std(dim=self.dim)
            inp = inp + torch.randn(inp.size()) * inp_std.unsqueeze(-1)*self.p_std + self.mean
        return inp.to(self.type)
    
    
class SaltAndPepper(torch.nn.Module):
    def __init__(self, probability: float = 1., noise: float = 0.1):
        '''
        Distorts an image with salt-and-pepper noise.

        Parameters
        ----------
        probability : float, optional
            Randomly adds noise to some of the elements of the input tensor with probability 
            using samples from a uniform distribution. The default is 1.0.
        noise : float, optional
            Probability of the salt and pepper noise. The default is 0.1.

        Returns
        -------
        None.

        '''
        super().__init__()
        self.probability = probability
        self.noise = noise
        
    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, noise={})'.format(
            self.probability, self.noise)
        
    def forward(self, img):
        """
        Distorts an image with salt-and-pepper noise.
        
        Parameters
        ----------
        img : torch.tensor
            C x W x H Tensor to add Salt and Pepper noise to.
            
        Returns
        -------
        img : torch.tensor
            Noisy Image.
        """
        if self.probability>torch.rand(1):
            noise = torch.rand_like(img)
            salt=torch.where(noise<self.noise)
            pepper=torch.where(noise>(1-self.noise))
            img[salt]=img.min().item()
            img[pepper]=img.max().item()
        return img
    
   
# %% additive


class AdditiveGaussianNoise(torch.nn.Module):
    """
    Parameters
    ----------
    amp : float, optional
        Amplitude of the added noise.
    """

    def __init__(self, amp):
        super(AdditiveGaussianNoise, self).__init__()
        self.Amplitude = amp

    def forward(self, ins):
        return ins+torch.randn_like(ins)*self.Amplitude

# %% multiplicative


class MultiplicativeGaussianNoise(torch.nn.Module):
    """
    Parameters
    ----------
    amp : float, optional
        Amplitude of the added noise.
    """

    def __init__(self, amp):
        super(MultiplicativeGaussianNoise, self).__init__()
        self.Amplitude = amp

    def forward(self, ins):
        return ins*(1+torch.randn_like(ins)*self.Amplitude)