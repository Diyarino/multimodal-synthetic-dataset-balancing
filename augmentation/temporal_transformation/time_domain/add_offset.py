# -*- coding: utf-8 -*-
"""
Random Bias / DC Offset Augmentation.

Adds a constant value to the signal or image.
- Audio/Signals: Simulates DC offset (sensor drift).
- Images: Simulates Brightness change.

@author: Diyar Altinses, M.Sc.
"""

import torch


class RandomBias(torch.nn.Module):
    """
    Adds a random constant offset to the input.
    Formula: Out = In + Uniform(low, high)
    
    Supports independent offsets per sample in a batch and per channel.
    """

    def __init__(
        self, 
        probability: float = 1.0, 
        range_val: float = 0.1, 
        per_channel: bool = False
    ):
        """
        Parameters
        ----------
        probability : float
            Probability of applying the augmentation.
        range_val : float
            Maximum offset magnitude. 
            The offset is chosen uniformly from [-range_val, +range_val].
            Example: 0.1 means adding a value between -0.1 and +0.1.
        per_channel : bool
            If True, generates different offsets for each channel (e.g. Color Shift).
            If False, adds the same offset to all channels (e.g. Brightness).
        """
        super().__init__()
        self.probability = probability
        self.range_val = range_val
        self.per_channel = per_channel

    def extra_repr(self) -> str:
        return f"probability={self.probability}, range={self.range_val}, per_channel={self.per_channel}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor. Shape (N, C, L) or (N, C, H, W).
            
        Returns:
            torch.Tensor: Offset tensor.
        """
        # 1. Probability Check
        if torch.rand(1, device=x.device) > self.probability:
            return x

        # 2. Determine Dimensions
        # We need to broadcast the offset: (N, C, 1, 1...)
        ndim = x.ndim
        
        # Determine Batch Size (N) and Channels (C)
        if ndim >= 2:
            N, C = x.shape[0], x.shape[1]
        else:
            # Fallback for single vectors without batch dim
            N, C = 1, 1 
            
        # 3. Generate Random Offsets
        # Shape generation depends on per_channel
        if self.per_channel:
            # Independent offset for every channel of every sample
            # Shape: (N, C)
            offsets = torch.empty(N, C, device=x.device).uniform_(-self.range_val, self.range_val)
        else:
            # Same offset for all channels of a sample (but different per sample)
            # Shape: (N, 1)
            offsets = torch.empty(N, 1, device=x.device).uniform_(-self.range_val, self.range_val)

        # 4. Reshape for Broadcasting
        # We need to append 1s for the spatial/time dimensions
        # If input is (N, C, L), we need (N, C, 1) or (N, 1, 1)
        # If input is (N, C, H, W), we need (N, C, 1, 1) or (N, 1, 1, 1)
        view_shape = list(offsets.shape) + [1] * (ndim - 2)
        offsets = offsets.view(*view_shape)

        # 5. Apply
        return x + offsets


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # ---------------------------------------------------------
    # Test 1: 1D Signals (Batch of 3)
    # Shape: (3, 1, 100)
    # ---------------------------------------------------------
    t = torch.linspace(0, 10, 100)
    # Create 3 identical sine waves
    batch_1d = torch.sin(t).view(1, 1, 100).repeat(3, 1, 1)
    
    # Augment: Different offsets per sample!
    aug_1d = RandomBias(probability=1.0, range_val=1.0, per_channel=False)
    out_1d = aug_1d(batch_1d.clone())

    # ---------------------------------------------------------
    # Test 2: 2D Images (Color Shift)
    # Shape: (1, 3, 100, 100) RGB
    # ---------------------------------------------------------
    img = torch.zeros(1, 3, 100, 100) # Black Image
    
    # Per-Channel: Offsets R, G, B differently -> Creates a colored tint
    aug_2d = RandomBias(probability=1.0, range_val=0.5, per_channel=True)
    out_2d = aug_2d(img.clone())

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    
    # 1D Plot
    for i in range(3):
        axs[0].plot(out_1d[i, 0].numpy(), label=f"Sample {i}")
    axs[0].plot(batch_1d[0, 0].numpy(), 'k--', label="Original", alpha=0.5)
    axs[0].set_title("1D: Independent DC Offsets per Sample")
    axs[0].legend()
    
    # 2D Plot
    # Clamp to display as image
    img_show = out_2d[0].permute(1, 2, 0).clamp(0, 1).numpy()
    axs[1].imshow(img_show)
    axs[1].set_title("2D: Per-Channel Bias (Color Tint)\n(Original was Black)")
    
    plt.tight_layout()
    plt.show()
    print("RandomBias test done.")