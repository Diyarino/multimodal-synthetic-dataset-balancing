# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 14:12:45 2022

@author: Diyar Altinses, M.Sc.

to-do:
    - 
"""

# %% imports

from typing import Tuple, Optional
import torch

# %% augmentations


class ChannelOffset(torch.nn.Module):
    """
    Adds a random offset to the channels of the input tensor.
    Useful for color jittering (RGB) or signal shifting.
    """
    
    def __init__(
        self, 
        probability: float = 1.0, 
        low: float = -1.0, 
        high: float = 1.0, 
        clamp: Optional[Tuple[float, float]] = None
    ):
        """
        Parameters
        ----------
        probability : float
            Probability of applying the transform (0.0 to 1.0).
        low : float
            Lower bound for the random offset.
        high : float
            Upper bound for the random offset.
        clamp : tuple, optional
            (min, max) to clamp the output values (e.g., 0.0 to 1.0 for images).
        """
        super().__init__()
        self.probability = probability
        self.low = low
        self.high = high
        self.clamp = clamp

    def extra_repr(self) -> str:
        """
        Standard PyTorch method to extend print representation.
        """
        return f'probability={self.probability}, low={self.low}, high={self.high}, clamp={self.clamp}'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies channel offset.
        
        Args:
            x (torch.Tensor): Input tensor of shape (C, H, W) or (N, C, H, W).
            
        Returns:
            torch.Tensor: Augmented tensor.
        """
        # 1. Probability check
        if torch.rand(1, device=x.device) > self.probability:
            return x

        # 2. Determine shape for noise generation
        # This logic ensures independent noise per image if a batch is provided.
        ndim = x.ndim
        if ndim == 3: # Format: (C, H, W)
            # Shape: (C, 1, 1) so it broadcasts over H and W
            noise_shape = (x.size(0), 1, 1)
        elif ndim == 4: # Format: (N, C, H, W)
            # Shape: (N, C, 1, 1) so every image in batch gets diff noise
            noise_shape = (x.size(0), x.size(1), 1, 1)
        else:
            # Fallback for 1D signals (N, C, L) or (C, L)
             noise_shape = x.shape[:-1] + (1,)

        # 3. Generate Noise on the correct device
        noise = torch.rand(noise_shape, device=x.device, dtype=x.dtype)
        
        # 4. Scale Noise: [0, 1] -> [low, high]
        noise = noise * (self.high - self.low) + self.low
        
        # 5. Apply
        out = x + noise

        # 6. Clamp if requested
        if self.clamp:
            out = torch.clamp(out, min=self.clamp[0], max=self.clamp[1])
            
        return out