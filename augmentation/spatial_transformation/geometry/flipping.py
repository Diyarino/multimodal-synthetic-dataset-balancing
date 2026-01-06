# -*- coding: utf-8 -*-
"""
Flipping Augmentation.

Randomly flips the input tensor along specified dimensions (e.g., horizontal or vertical flip).
Safe for batch processing.

@author: Diyar Altinses, M.Sc.
"""

from typing import Tuple, Union
import torch


class RandomFlip(torch.nn.Module):
    """
    Randomly reverses the order of elements along the given dimensions.
    Useful for Horizontal Flip (Left-Right) or Vertical Flip (Up-Down).
    """

    def __init__(self, probability: float = 0.5, dims: Union[int, Tuple[int, ...]] = (-1,)):
        """
        Parameters
        ----------
        probability : float
            Probability of applying the flip.
        dims : int or tuple of ints
            The dimensions to flip. 
            Standard for images (..., H, W):
            - Horizontal Flip: dim = -1 (Width)
            - Vertical Flip:   dim = -2 (Height)
            - Both:            dims = (-2, -1)
        """
        super().__init__()
        self.probability = probability
        # Ensure dims is a tuple for consistency
        self.dims = (dims,) if isinstance(dims, int) else dims

    def extra_repr(self) -> str:
        return f"probability={self.probability}, dims={self.dims}"

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): Input image. Shape (C, H, W) or (N, C, H, W).
            
        Returns:
            torch.Tensor: Flipped image.
        """
        # 1. Probability Check
        if torch.rand(1, device=img.device) > self.probability:
            return img

        # 2. Apply Flip
        # torch.flip handles arbitrary dimensions safely.
        # Using negative indices (-1, -2) is safer than (2, 3) because it works
        # for both (C, H, W) and (N, C, H, W) inputs.
        
        return torch.flip(img, self.dims)


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # Create an asymmetric image (gradient) to clearly see flipping
    # Shape: (1, 1, 100, 100) -> Batch=1, Channel=1
    H, W = 100, 100
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    img = (x + y).float().unsqueeze(0).unsqueeze(0)
    img = img / img.max()
    
    # 1. Horizontal Flip (Left-Right)
    # dim = -1 means the last dimension (Width)
    aug_h = RandomFlip(probability=1.0, dims=(-1,))
    out_h = aug_h(img)

    # 2. Vertical Flip (Up-Down)
    # dim = -2 means the second to last dimension (Height)
    aug_v = RandomFlip(probability=1.0, dims=(-2,))
    out_v = aug_v(img)

    # Visualization
    

    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    
    axs[0].imshow(img[0, 0], cmap='viridis')
    axs[0].set_title("Original\n(Dark Top-Left)")
    axs[0].axis('off')
    
    axs[1].imshow(out_h[0, 0], cmap='viridis')
    axs[1].set_title("Horizontal Flip\n(Dark Top-Right)")
    axs[1].axis('off')

    axs[2].imshow(out_v[0, 0], cmap='viridis')
    axs[2].set_title("Vertical Flip\n(Dark Bottom-Left)")
    axs[2].axis('off')
    
    plt.tight_layout()
    plt.show()