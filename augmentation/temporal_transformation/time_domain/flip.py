# -*- coding: utf-8 -*-
"""
Flipping & Reversal Augmentations.

Geometric transformations that reverse the order of elements along specific axes.
1. RandomFlip: General purpose flipping (Horizontal, Vertical).
2. TimeReversal: Specifically for Time Series (reading signal backwards).
3. RandomNegate: Inverts the values (Signal flipping upside down).

@author: Diyar Altinses, M.Sc.
"""

from typing import Tuple, Union
import torch


class RandomFlip(torch.nn.Module):
    """
    Randomly flips the input tensor along specified dimensions.
    
    - For Images (H, W): dim=(-1,) is Horizontal Flip, dim=(-2,) is Vertical Flip.
    - For Signals (L): dim=(-1,) is Time Reversal.
    """

    def __init__(self, probability: float = 0.5, dims: Union[int, Tuple[int, ...]] = (-1,)):
        """
        Parameters
        ----------
        probability : float
            Probability of flipping.
        dims : int or tuple
            The dimension(s) to flip. 
            Default (-1) flips the last dimension (Width or Time).
        """
        super().__init__()
        self.probability = probability
        self.dims = dims if isinstance(dims, (tuple, list)) else (dims,)

    def extra_repr(self) -> str:
        return f"probability={self.probability}, dims={self.dims}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor.
        """
        # 1. Probability Check
        if torch.rand(1, device=x.device) > self.probability:
            return x

        # 2. Apply Flip
        return torch.flip(x, dims=self.dims)


class TimeReversal(torch.nn.Module):
    """
    Reverses the time axis (reads the signal backwards).
    Specific alias for RandomFlip(dim=-1).
    """

    def __init__(self, probability: float = 0.5):
        super().__init__()
        self.probability = probability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1, device=x.device) > self.probability:
            return x
        
        # Flip only the last dimension (Time/Length)
        return torch.flip(x, dims=(-1,))


class RandomNegate(torch.nn.Module):
    """
    Inverts the signal values (Value Flipping).
    Mathematically: x = x * -1
    
    (Formerly named 'BiDirectionalFlipping_channelwise' in original code, which was misleading).
    """

    def __init__(self, probability: float = 0.5):
        super().__init__()
        self.probability = probability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1, device=x.device) > self.probability:
            return x
            
        return -x


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # ---------------------------------------------------------
    # Test Data: Asymmetrical Signal (Sawtooth)
    # Shape: (1, 1, 100)
    # ---------------------------------------------------------
    t = torch.linspace(0, 1, 100)
    # Sawtooth wave: Rises slowly, drops instantly
    signal = (t % 1.0).view(1, 1, 100)
    
    # ---------------------------------------------------------
    # 1. Time Reversal (Horizontal Flip)
    # ---------------------------------------------------------
    aug_time = TimeReversal(probability=1.0)
    out_time = aug_time(signal.clone())

    # ---------------------------------------------------------
    # 2. Value Negation (Vertical Flip of the graph)
    # ---------------------------------------------------------
    aug_neg = RandomNegate(probability=1.0)
    out_neg = aug_neg(signal.clone())

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------
    

    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    
    # Original
    axs[0].plot(signal[0, 0].numpy(), 'k')
    axs[0].set_title("Original (Sawtooth)")
    axs[0].grid(True, alpha=0.3)
    
    # Time Reversal
    axs[1].plot(out_time[0, 0].numpy(), 'b')
    axs[1].set_title("Time Reversal\n(Flipped Left-Right)")
    axs[1].grid(True, alpha=0.3)
    
    # Negation
    axs[2].plot(out_neg[0, 0].numpy(), 'r')
    axs[2].set_title("Negation\n(Flipped Up-Down)")
    axs[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    print("Flipping Augmentations test done.")