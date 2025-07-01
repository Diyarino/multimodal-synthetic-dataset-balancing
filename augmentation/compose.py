# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 14:11:59 2022

@author: Diyar Altinses, M.Sc.

to-do:
    - 
"""

#%% imports

import torch
import augmentation as augmentations
import torchvision.transforms as transforms

# %% augmentations

class Compose(torch.nn.Module):
    def __init__(self, augmentation_dict: dict = {}, shuffle: bool = False, mode = 'single'):
        """
        Load the Modules of the augmenttaion dict with the corresponding arguments and compose them.
    
        Parameters
        ----------
        inp : array
            
        augmentation_dict : dict, optional
            The augmentation methods with the corresponding arguments. The default is {}.
        shuffle : bool, optional
            If the augmentation dict should be shuffled. The default is False.
        mode : str, optional
            Defines weather one random augmnetation is applied or all.
            
        Returns
        -------
        None.
        
        """
        super().__init__()
        self.augmentation_dict = augmentation_dict
        self.shuffle = shuffle
        self.mode = mode
        self.modules = self.load_modules()
        self.augmentation = transforms.Compose(self.modules)
        
    def load_modules(self):
        '''
        Load all augmentations from costum lib or torch lib. if its available in torch it is prefered.

        Raises
        ------
            Raise error if modules doesnt exist in both of the libs.

        Returns
        -------
        torchvision.transforms
            The whole augmentations composed in one object.

        '''
        modules = []
        keys = list(self.augmentation_dict)
        index = torch.randperm(len(keys)) if self.shuffle else torch.arange(len(keys))
        for idx in index:
            if hasattr(transforms, keys[idx]):
                modules.append(getattr(transforms, keys[idx])(**self.augmentation_dict[keys[idx]]))
            elif hasattr(augmentations, keys[idx]):
                modules.append(getattr(augmentations, keys[idx])(**self.augmentation_dict[keys[idx]]))
            else:
                raise Exception('do not know this augmentation method: {}'.format(keys[idx]))

        return modules
    
    def select_single(self, idx: int = 0):
        '''
        Select one single augmentation from the dict and use it.

        Parameters
        ----------
        idx : int
            The index of the selected augmentation.

        Returns
        -------
        None.

        '''
        #self.modules = [self.modules[idx]]
        self.augmentation = transforms.Compose([self.modules[idx]])
        return self.modules[idx]
        
        
    def select_combinations(self, combinations: list = []):
        '''
        Select one multiple augmentation from the dict and use it.

        Parameters
        ----------
        combinations : list[List[idx,...]]
            The indexes of the selected augmentation.

        Returns
        -------
        None.

        '''
        combined_modules = []
        for combi in combinations:
            modules = [self.modules[idx] for idx in combi]
            combined_modules.append(transforms.Compose(modules))
        self.modules = combined_modules
        self.augmentation = transforms.Compose(combined_modules)
    
    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return '{}'.format(self.augmentation)
        
    def forward(self, inp: torch.Tensor, batched = False) -> torch.Tensor:
        """
        Call argumentation.

        Parameters
        ----------
        inp : torch.Tensor
            The input array which needs to be flipped.

        Returns
        -------
        out :  torch.Tensor
            The modified input.

        """
        if not batched:
            if self.mode == 'single':
                index = torch.randint(0, len(self.modules), (1,))
                self.select_single(index)
                out = self.augmentation(inp)
            else:
                out = self.augmentation(inp)
                
        if batched:
            storage = []
            for sample in inp:
                index = torch.randint(0, len(self.modules), (1,))
                self.select_single(index)
                storage.append(self.augmentation(sample))
            out = torch.stack(storage)
            
        return out
    
    
    
# %% augmentations


class RandomApplyArea(torch.nn.Module):
    def __init__(self, start: tuple = (0,0), len: int = 1, width: int = 1, 
                 random: bool = False, augmentation_dict: dict = {}):
        '''
        

        Parameters
        ----------
        start : tuple, optional
            DESCRIPTION. The default is (0,0).
        len : int, optional
            DESCRIPTION. The default is 1.
        width : int, optional
            DESCRIPTION. The default is 1.
        random : bool, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        '''
        super().__init__()
        self.start = start
        self.len = len
        self.width = width
        self.random = random
        self.augmentation = Compose(augmentation_dict)
        
    def __repr__(self):
        """
        Represent the class.

        Returns
        -------
        str
            The representation of the class.

        """
        return self.__class__.__name__+'(start={}, len={}, width={}, random={})'.format(
            self.start, self.len, self.width, self.random)

    def forward(self, inp: torch.Tensor, min_size: int = 10) -> torch.Tensor:
        """
        Call argumentation.

        Parameters
        ----------
        inp : torch.Tensor
            The input array which needs to be puzzled. Have to be 1 or 2 (batch) dimensional.

        Returns
        -------
        inp :  torch.Tensor
            The modified input.

        """
        if self.random:
            shape = torch.tensor(inp.shape)
            self.start = torch.randint(0, min(shape[-2:] - min_size), (2,))
            self.len = torch.randint(0, shape[-2] - self.start[0], (1,))
            self.width = torch.randint(0, shape[-1] - self.start[1], (1,))
        inp[..., self.start[0]: self.start[0] + self.len, self.start[1]: self.start[1] + self.width] = self.augmentation(
            inp[..., self.start[0]: self.start[0] + self.len, self.start[1]: self.start[1] + self.width])
        return inp
    
    
    
    
    
    
    
    
    
    
    
    
    
    