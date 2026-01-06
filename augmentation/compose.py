# -*- coding: utf-8 -*-
"""
Augmentation Pipeline and Region Processing.

Provides classes to compose multiple augmentations and apply them 
either globally or to specific regions of the data.

@author: Diyar Altinses, M.Sc.
"""

from typing import Dict, List, Tuple, Union, Optional, Any
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import augmentation as augmentations


class Compose(nn.Module):
    """
    Composes multiple augmentations based on a configuration dictionary.
    
    Modes:
    - 'all': Applies all defined augmentations sequentially.
    - 'single': Selects exactly one augmentation randomly per forward pass.
    """

    def __init__(
        self, 
        augmentation_dict: Dict[str, dict], 
        shuffle: bool = False, 
        mode: str = 'single'
    ):
        """
        Parameters
        ----------
        augmentation_dict : dict
            Dictionary where keys are class names and values are parameter dicts.
            Example: {'AdditiveGaussianNoise': {'std': 0.1}, 'RandomFlip': {}}
        shuffle : bool
            If True, shuffles the order of augmentations (only relevant for mode='all').
        mode : str
            'all' or 'single'. Defines execution strategy.
        """
        super().__init__()
        self.config = augmentation_dict
        self.shuffle = shuffle
        self.mode = mode
        
        # Load modules and register them as sub-modules
        self.ops = nn.ModuleList(self._load_modules())

    def _load_modules(self) -> List[nn.Module]:
        """
        Dynamically loads augmentation classes from 'augmentation' lib or 'torchvision'.
        """
        modules = []
        for name, params in self.config.items():
            # 1. Try Custom Lib (Your optimized classes)
            if hasattr(augmentations, name):
                cls = getattr(augmentations, name)
                modules.append(cls(**params))
            
            # 2. Try Torchvision
            elif hasattr(transforms, name):
                cls = getattr(transforms, name)
                modules.append(cls(**params))
            
            else:
                raise ImportError(f"Augmentation class '{name}' not found in 'augmentation' or 'torchvision'.")
        
        return modules

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentations.

        Args:
            x (torch.Tensor): Input tensor.
        """
        if len(self.ops) == 0:
            return x

        # Determine execution order
        indices = list(range(len(self.ops)))

        if self.mode == 'single':
            # Pick one random index
            idx = torch.randint(0, len(self.ops), (1,)).item()
            indices = [idx]
        elif self.shuffle:
            # Random permutation of all indices
            indices = torch.randperm(len(self.ops)).tolist()

        # Execute
        for i in indices:
            x = self.ops[i](x)

        return x

    def __repr__(self):
        return f"AugmentationPipeline(mode='{self.mode}', ops={self.ops})"


class RandomApplyArea(nn.Module):
    """
    Applies an augmentation pipeline to a random rectangular sub-region.
    (Formerly RandomApplyArea).
    """

    def __init__(
        self, 
        augmentation_dict: Dict[str, dict],
        min_size: Tuple[int, int] = (10, 10),
        random_pos: bool = True,
        fixed_region: Optional[Tuple[int, int, int, int]] = None
    ):
        """
        Parameters
        ----------
        augmentation_dict : dict
            Configuration for the augmentations to apply inside the region.
        min_size : tuple
            (min_height, min_width) for the random region.
        random_pos : bool
            If True, selects a new random region for every batch/forward pass.
        fixed_region : tuple, optional
            (start_y, start_x, height, width) used if random_pos is False.
        """
        super().__init__()
        self.pipeline = Compose(augmentation_dict, mode='all')
        self.min_size = min_size
        self.random_pos = random_pos
        self.fixed_region = fixed_region

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input (N, C, H, W) or (N, C, L).
        """
        # Handle 3D inputs (N, C, L) by viewing as 4D (N, C, 1, L)
        is_1d = False
        if x.ndim == 3:
            is_1d = True
            x = x.unsqueeze(2)

        N, C, H, W = x.shape

        # Determine Region Parameters
        if self.random_pos:
            # Ensure valid range
            max_h = max(H, self.min_size[0])
            max_w = max(W, self.min_size[1])

            # Random size
            h_len = torch.randint(self.min_size[0], max_h + 1, (1,)).item()
            w_len = torch.randint(self.min_size[1], max_w + 1, (1,)).item()
            
            # Clamp actual size to image bounds
            h_len = min(h_len, H)
            w_len = min(w_len, W)

            # Random start
            y_start = torch.randint(0, H - h_len + 1, (1,)).item()
            x_start = torch.randint(0, W - w_len + 1, (1,)).item()
        else:
            if self.fixed_region is None:
                return x if not is_1d else x.squeeze(2)
            y_start, x_start, h_len, w_len = self.fixed_region

        # 1. Extract Region
        # Clone to avoid in-place modification errors in computation graph
        out = x.clone()
        region = out[..., y_start : y_start + h_len, x_start : x_start + w_len]

        # 2. Apply Pipeline to Region
        # If is_1d, we need to handle squeezing inside the pipeline if the pipeline expects 3D
        # But our optimized augmentations handle (N, C, H, W) fine usually.
        augmented_region = self.pipeline(region)

        # 3. Insert back
        out[..., y_start : y_start + h_len, x_start : x_start + w_len] = augmented_region

        if is_1d:
            out = out.squeeze(2)

        return out