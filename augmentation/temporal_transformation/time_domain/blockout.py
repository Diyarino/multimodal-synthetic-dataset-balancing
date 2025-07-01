# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 14:58:15 2022

@author: Diyar Altinses, M.Sc.

to-do:
    - 
"""
# %% imports

import torch

# %% classes

class BlockOut(torch.nn.Module):
    def __init__(self, probability: float = 1., block_size: tuple = (0,0)):
        """
        Randomly apply blockout to time series data.

        Parameters
        ----------
        probability : float, optional
            Apply argumenttaion with probability using samples from a uniform distribution. The default is 1.0.
        block_size : int, optional
            Define the size of the blockout
            
        Returns
        -------
        None.

        """
        super().__init__()
        self.probability = probability
        self.block_size = block_size

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
            
            size = inp.size()
            
            pos = torch.randint(0, size[-1]-self.block_size[1],(1,))
            len_ = torch.randint(self.block_size[0], self.block_size[1],(1,))

            inp[..., pos:pos+len_] = 0
            
        return inp

# %% test

if __name__ == '__main__':
	test = torch.rand(1)