# -*- coding: utf-8 -*-%
"""
Created on %(date)s

@author: Diyar Altinses, M.Sc.

to-do:
    - 
"""

# %% imports

import torch

# %% application

class Nebula(torch.nn.Module):
    def __init__(self, probability: float = .5, visibility_range: float = 512., distance: float = 256., 
                 luminance: float = 256., type_: str = 'int32'):
        '''
        This class add some fog to an image using the koschmieders model. This model designed for images
        with depth information but als works more or less with usual images.

        Parameters
        ----------
        probability : float, optional
            Randomly adds noise to some of the elements of the input tensor with probability 
            using samples from a uniform distribution. The default is 0.5.
        visibility_range : float, optional
            The visibility range of the image. If range is smaller then distance -> very foggy. The default is 512..
        distance : float, optional
            The distance to the obstacle. The default is 256.
        luminance : float, optional
            The luminance of the sky in the image. The default is 256.
        type_ : str, optional
                    The target type of the data. torch dtype class name. The default is int32.

        Returns
        -------
        None.

        '''
        super().__init__()
        
        self.probability = probability
        self.visibility_range = visibility_range
        self.distance = distance
        self.luminance = luminance
        self.type = getattr(torch, type_)
        
    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(probability={}, visibility_range={}, distance={}, luminance={})'.format(
            self.probability, self.visibility_range, self.distance, self.luminance)
    
    def forward(self, inp: torch.Tensor):
        """
        Call argumentation.
    
        Parameters
        ----------
        inp : torch.Tensor
            The input array which needs some fog.
    
        Returns
        -------
        inp :  torch.Tensor
            The modified input.
    
        """
        if torch.rand(1) <= self.probability:
            channel, height, width = inp.shape
            distance = torch.zeros_like(inp) + torch.linspace(self.distance,0, height).expand(-1,width)
            kd = torch.log(torch.tensor(20))/self.visibility_range*distance
            inp = inp*torch.exp(-kd)+self.luminance*(1-torch.exp(-kd))
        return inp.to(self.type)

    
