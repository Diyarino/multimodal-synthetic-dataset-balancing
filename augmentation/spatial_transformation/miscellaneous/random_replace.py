# -*- coding: utf-8 -*-
"""
Random Constant Replacement Augmentation.

Replaces the entire image/signal with a constant value (e.g. 0 for blackout).
Simulates total sensor failure, missing data, or blank frames.

@author: Diyar Altinses, M.Sc.
"""

from typing import Tuple
import torch


class RandomConstantReplace(torch.nn.Module):
    """
    Randomly replaces the input tensor with a constant value chosen from a list.
    Useful for simulating 'Dead Signals', 'Black Frames', or 'White Screens'.
    """

    def __init__(
        self, 
        probability: float = 1.0, 
        values: Tuple[float, ...] = (0.0, 1.0)
    ):
        """
        Parameters
        ----------
        probability : float
            Probability of replacing the input.
        values : tuple of floats
            List of possible values to replace the input with.
            Example: (0.0,) -> Always replace with black.
            Example: (0.0, 1.0) -> Randomly replace with black OR white.
        """
        super().__init__()
        self.probability = probability
        self.values = values

    def extra_repr(self) -> str:
        return f"probability={self.probability}, possible_values={self.values}"

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): Input image (C, H, W) or (N, C, H, W).
            
        Returns:
            torch.Tensor: Original image OR Constant tensor.
        """
        # 1. Probability Check (Global early exit if prob=0)
        if self.probability == 0:
            return img

        device = img.device
        
        # 2. Setup Dimensions
        is_batch = img.ndim == 4
        x = img if is_batch else img.unsqueeze(0)
        N, C, H, W = x.shape

        # 3. Determine which samples to replace
        # Shape: (N, 1, 1, 1) for broadcasting
        # Generate random floats [0, 1] and compare to probability
        mask = torch.rand((N, 1, 1, 1), device=device) < self.probability
        
        # If no images need replacement, return early
        if not mask.any():
            return img

        # 4. Select Replacement Values
        # We want to pick a random value from self.values for EACH image.
        
        # Convert values list to tensor on device
        candidate_values = torch.tensor(self.values, device=device, dtype=x.dtype)
        
        # Generate random indices: (N,)
        indices = torch.randint(0, len(candidate_values), (N,), device=device)
        
        # Gather chosen values: (N,) -> Reshape to (N, 1, 1, 1)
        chosen_vals = candidate_values[indices].view(N, 1, 1, 1)

        # 5. Apply Replacement
        # torch.where(condition, x, y) -> Where mask is True, use chosen_vals. Else use original x.
        # Broadcasting expands chosen_vals (N, 1, 1, 1) to (N, C, H, W)
        out = torch.where(mask, chosen_vals, x)

        return out if is_batch else out.squeeze(0)


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # Create a batch of 3 gradient images
    # Shape: (3, 1, 100, 100)
    H, W = 100, 100
    grad = torch.linspace(0, 1, W).view(1, W).expand(H, W).unsqueeze(0).unsqueeze(0)
    batch = grad.repeat(3, 1, 1, 1)
    
    # Init Augmentation
    # Probability 1.0 means ALL will be replaced.
    # We choose between 0.0 (Black) and 1.0 (White) and 0.5 (Gray) randomly.
    aug = RandomConstantReplace(probability=1.0, values=(0.0, 0.5, 1.0))
    
    out = aug(batch)

    # Visualization
    fig, axs = plt.subplots(2, 3, figsize=(10, 6))
    
    # Row 1: Originals
    for i in range(3):
        axs[0, i].imshow(batch[i, 0], cmap='gray', vmin=0, vmax=1)
        axs[0, i].set_title(f"Original {i}")
        axs[0, i].axis('off')
        
    # Row 2: Augmented (Should be solid colors)
    for i in range(3):
        axs[1, i].imshow(out[i, 0], cmap='gray', vmin=0, vmax=1)
        val = out[i, 0, 0, 0].item()
        axs[1, i].set_title(f"Replaced {i}\nVal: {val:.1f}")
        axs[1, i].axis('off')
    
    plt.suptitle("RandomConstantReplace Test (Simulating Signal Loss)")
    plt.tight_layout()
    plt.show()
    print("RandomConstantReplace test done.")