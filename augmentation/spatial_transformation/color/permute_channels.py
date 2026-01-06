# -*- coding: utf-8 -*-
"""
Channel Permutation Augmentation.

Randomly shuffles the order of channels (e.g., RGB -> BGR, or swapping sensor axes).
Useful for creating invariance against color/sensor ordering.

@author: Diyar Altinses, M.Sc.
"""

from typing import Optional
import torch


class PermuteChannels(torch.nn.Module):
    """
    Randomly permutes the channels of the input tensor.
    Example: Converts RGB to BGR, GRB, etc.
    """

    def __init__(self, probability: float = 0.5, channel_dim: Optional[int] = None):
        """
        Parameters
        ----------
        probability : float
            Probability of applying the permutation.
        channel_dim : int, optional
            The dimension index of the channels.
            If None (default), it is auto-detected:
                - Dim 0 for 3D tensors (C, H, W)
                - Dim 1 for 4D tensors (N, C, H, W)
        """
        super().__init__()
        self.probability = probability
        self.channel_dim = channel_dim

    def extra_repr(self) -> str:
        return f"probability={self.probability}, channel_dim={self.channel_dim if self.channel_dim is not None else 'Auto'}"

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): Input image. Shape (C, H, W) or (N, C, H, W).
            
        Returns:
            torch.Tensor: Image with permuted channels.
        """
        # 1. Probability Check
        if torch.rand(1, device=img.device) > self.probability:
            return img

        # 2. Determine Channel Dimension (Auto-Detect logic)
        if self.channel_dim is not None:
            dim = self.channel_dim
        else:
            # Standard PyTorch layout assumption:
            # 3D -> (C, H, W) -> Dim 0
            # 4D -> (N, C, H, W) -> Dim 1
            dim = 1 if img.ndim == 4 else 0

        # 3. Create Permutation Indices
        num_channels = img.shape[dim]
        if num_channels <= 1:
            return img  # Nothing to permute for grayscale

        # Generate random permutation on the correct device
        perm_indices = torch.randperm(num_channels, device=img.device)

        # 4. Apply Permutation efficiently
        # index_select is generally faster and cleaner than manual slicing for this task
        out = torch.index_select(img, dim, perm_indices)

        return out


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # Create a dummy image with distinct colors per channel to visualize swaps
    # Shape: (3, 100, 100) -> R, G, B
    H, W = 100, 100
    img = torch.zeros(3, H, W)
    
    # Fill channels with patterns
    # Channel 0 (Red): Horizontal bars
    img[0, 20:40, :] = 1.0 
    # Channel 1 (Green): Vertical bars
    img[1, :, 40:60] = 1.0 
    # Channel 2 (Blue): Full block in corner
    img[2, 60:, 60:] = 1.0 
    
    # Init Augmentation
    aug = PermuteChannels(probability=1.0) # Force permute for test

    # Apply multiple times to see different permutations
    out1 = aug(img.clone())
    out2 = aug(img.clone())

    # Visualization
    

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    
    axs[0].imshow(img.permute(1, 2, 0))
    axs[0].set_title("Original (RGB)")
    axs[0].axis('off')

    axs[1].imshow(out1.permute(1, 2, 0))
    axs[1].set_title("Permutation 1")
    axs[1].axis('off')

    axs[2].imshow(out2.permute(1, 2, 0))
    axs[2].set_title("Permutation 2")
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()
    print("PermuteChannels test completed.")