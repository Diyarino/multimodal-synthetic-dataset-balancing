# -*- coding: utf-8 -*-
"""
Augmentation Classes.

Specific augmentation implementations compatible with PyTorch pipelines.

@author: Diyar Altinses, M.Sc.
"""

from typing import Optional, Tuple
import torch

# %% augmentations

class ColorOffset(torch.nn.Module):
    """
    Adds a random offset to all channels of the input tensor uniformly.
    This effectively changes the brightness/intensity of the image or signal.
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
        clamp : tuple of float, optional
            If provided, clamps the output to (min, max). 
            Example: (0.0, 1.0) for standard normalized images.
        """
        super().__init__()
        self.probability = probability
        self.low = low
        self.high = high
        self.clamp = clamp

    def extra_repr(self) -> str:
        """Standard PyTorch representation."""
        return f'probability={self.probability}, low={self.low}, high={self.high}, clamp={self.clamp}'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the brightness offset.

        Args:
            x (torch.Tensor): Input tensor. 
                              Shape (C, H, W) for single image 
                              or (N, C, H, W) for batch.

        Returns:
            torch.Tensor: Modified tensor.
        """
        # 1. Check Probability (on correct device)
        if torch.rand(1, device=x.device) > self.probability:
            return x

        # 2. Determine Shape for independent noise per image
        ndim = x.ndim
        if ndim == 4: 
            # Batch input: (N, C, H, W) -> Noise shape (N, 1, 1, 1)
            # Each image gets a DIFFERENT offset, but same offset across C,H,W
            noise_shape = (x.size(0), 1, 1, 1)
        else:
            # Single input: (C, H, W) -> Noise shape (1, 1, 1)
            noise_shape = (1, 1, 1)

        # 3. Generate Noise on correct device
        # Formula: rand * (max - min) + min
        noise = torch.rand(noise_shape, device=x.device, dtype=x.dtype)
        offset = noise * (self.high - self.low) + self.low

        # 4. Apply
        out = x + offset

        # 5. Clamp (optional but recommended for images)
        if self.clamp:
            out = torch.clamp(out, min=self.clamp[0], max=self.clamp[1])

        return out
