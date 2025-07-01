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
        if self.probability > torch.rand(1):
            noise = torch.rand_like(img)
            salt = torch.where(noise < self.noise)
            pepper = torch.where(noise > (1-self.noise))
            img[salt] = img.min().item()
            img[pepper] = img.max().item()
        return img
