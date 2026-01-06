# -*- coding: utf-8 -*-
"""
Time Cutout / BlockOut Augmentation.

Randomly masks out a contiguous section of the time series.
Forces the model to learn features from the context rather than relying on a specific local segment.

@author: Diyar Altinses, M.Sc.
"""

from typing import Tuple, Union
import torch


class TimeCutout(torch.nn.Module):
    """
    Randomly zeroes out a contiguous chunk of the time series.
    (Also known as 'BlockOut' or 'Erasure').
    """

    def __init__(
        self, 
        probability: float = 1.0, 
        block_size: Tuple[int, int] = (10, 50), 
        fill_value: float = 0.0
    ):
        """
        Parameters
        ----------
        probability : float
            Probability of applying the augmentation.
        block_size : tuple (min, max)
            Range for the size of the hole to cut out (in time steps).
            Example: (10, 50) cuts a hole between 10 and 50 steps long.
        fill_value : float
            Value to fill the hole with (default 0.0).
            Can be set to None (TODO) for noise fill, but usually 0 is standard.
        """
        super().__init__()
        self.probability = probability
        self.min_len = block_size[0]
        self.max_len = block_size[1]
        self.fill_value = fill_value

    def extra_repr(self) -> str:
        return f"probability={self.probability}, block_range=({self.min_len}, {self.max_len}), fill={self.fill_value}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor. Shape (N, C, L).
            
        Returns:
            torch.Tensor: Augmented tensor with holes.
        """
        # 1. Probability Check
        if torch.rand(1, device=x.device) > self.probability:
            return x

        # 2. Dimensions
        # Ensure input is at least 3D (N, C, L)
        if x.ndim == 2:
            x = x.unsqueeze(1)
            
        N, C, L = x.shape
        device = x.device

        # 3. Generate Random Parameters per Sample (Vectorized)
        
        # Determine lengths for each sample: (N,)
        # Clamp max_len to L to avoid errors if signal is short
        safe_max_len = min(self.max_len, L)
        safe_min_len = min(self.min_len, safe_max_len)
        
        # Lengths between min and max
        lengths = torch.randint(safe_min_len, safe_max_len + 1, (N,), device=device)
        
        # Determine start positions for each sample: (N,)
        # Start must be between 0 and L - length
        # Since lengths vary per sample, calculating exact max_start per sample is tricky vectorized.
        # Efficient trick: Generate random float [0, 1] and scale by (L - length)
        rand_factors = torch.rand(N, device=device)
        starts = (rand_factors * (L - lengths)).long()

        # 4. Create Mask (Vectorized)
        # We create a coordinate grid [0, 1, 2, ..., L-1]
        # And compare: start <= grid < start + length
        
        # Grid: (1, L)
        seq_grid = torch.arange(L, device=device).unsqueeze(0)
        
        # Expand starts/lengths to (N, 1)
        starts_exp = starts.unsqueeze(1)
        lengths_exp = lengths.unsqueeze(1)
        
        # Boolean Mask: (N, L) -> True where we want to cut
        mask = (seq_grid >= starts_exp) & (seq_grid < (starts_exp + lengths_exp))
        
        # Reshape mask to (N, 1, L) to broadcast over channels
        mask = mask.unsqueeze(1)

        # 5. Apply Cutout
        # Use clone to be safe (out-of-place)
        out = x.clone()
        
        # Where mask is True, set to fill_value
        out.masked_fill_(mask, self.fill_value)

        return out


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # ---------------------------------------------------------
    # Test: Batch of 3 Signals
    # Shape: (3, 1, 100)
    # ---------------------------------------------------------
    t = torch.linspace(0, 10, 100)
    # Create sine waves
    batch = torch.sin(t).view(1, 1, 100).repeat(3, 1, 1)
    
    # Augment: Cut random holes of length 15 to 30
    aug = TimeCutout(probability=1.0, block_size=(15, 30), fill_value=0.0)
    out = aug(batch)

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------

    fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True, sharey=True)
    
    colors = ['blue', 'orange', 'green']
    
    for i in range(3):
        axs[i].plot(t.numpy(), batch[i, 0].numpy(), 'k--', alpha=0.3, label="Original")
        axs[i].plot(t.numpy(), out[i, 0].numpy(), color=colors[i], label="Augmented")
        
        # Highlight the cut (where value is 0.0 but original wasn't close to 0)
        # Just for visualization
        axs[i].set_title(f"Sample {i}: Random Cutout")
        axs[i].legend(loc='upper right')
        axs[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    print("TimeCutout test done.")