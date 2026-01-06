# -*- coding: utf-8 -*-
"""
Waterdrops Augmentation.

Simulates water drops on the lens by blurring circular regions.
Uses Gaussian Blur and mask accumulation for efficiency.

@author: Diyar Altinses, M.Sc.
"""

from typing import Tuple
import torch
from torchvision.transforms import GaussianBlur

class Waterdrops(torch.nn.Module):
    """
    Adds "water drops" to the image. 
    Technically, it creates a blurred version of the image and selectively 
    reveals it through circular masks to simulate out-of-focus drops on a lens.
    """

    def __init__(
        self, 
        probability: float = 0.5, 
        amount: int = 10, 
        radius_max: int = 50,
        kernel_size: Tuple[int, int] = (25, 25), 
        sigma: Tuple[float, float] = (5.0, 10.0)
    ):
        """
        Parameters
        ----------
        probability : float
            Probability of applying the augmentation.
        amount : int
            Maximum number of drops to generate.
        radius_max : int
            Maximum radius of a drop in pixels.
        kernel_size : tuple
            Size of the Gaussian kernel (must be odd numbers).
        sigma : tuple
            Range (min, max) for the standard deviation of the blur.
        """
        super().__init__()
        self.probability = probability
        self.amount = amount
        self.radius_max = radius_max
        
        # Ensure kernel sizes are odd numbers (required by PyTorch)
        k_h, k_w = kernel_size
        if k_h % 2 == 0: k_h += 1
        if k_w % 2 == 0: k_w += 1
        
        self.blur = GaussianBlur(kernel_size=(k_h, k_w), sigma=sigma)

    def extra_repr(self) -> str:
        return f"probability={self.probability}, amount={self.amount}, radius_max={self.radius_max}"

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): Input image (C, H, W) or (N, C, H, W).
            
        Returns:
            torch.Tensor: Image with blurred water spots.
        """
        # 1. Probability Check
        if torch.rand(1, device=img.device) > self.probability:
            return img

        # 2. Setup Dimensions
        is_batch = img.ndim == 4
        x = img if is_batch else img.unsqueeze(0)
        N, C, H, W = x.shape
        device = x.device

        # 3. Create Blurred Version (Once per batch)
        # This is the "content" of the water drops
        blurred_x = self.blur(x)

        # 4. Create Accumulation Mask
        # Shape: (N, 1, H, W) to broadcast over channels
        total_mask = torch.zeros((N, 1, H, W), device=device, dtype=torch.bool)

        # 5. Generate Grid (Once)
        # indexing='xy' means first dim is X (width), second is Y (height)
        # We use 'ij' to match matrix notation (Height, Width)
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )

        # 6. Loop to generate drops
        # We loop 'amount' times, but operations inside are vectorized over Batch N
        # This keeps memory usage low compared to creating a (N, Amount, H, W) tensor.
        
        # Pre-generate random parameters for all iterations
        # Centers (N, amount)
        center_x = torch.randint(0, W, (N, self.amount), device=device).float()
        center_y = torch.randint(0, H, (N, self.amount), device=device).float()
        radii = torch.randint(1, self.radius_max + 1, (N, self.amount), device=device).float()
        radii_sq = radii ** 2

        for i in range(self.amount):
            # Extract params for the i-th drop across the batch
            cx = center_x[:, i].view(N, 1, 1)
            cy = center_y[:, i].view(N, 1, 1)
            r2 = radii_sq[:, i].view(N, 1, 1)

            # Calculate distance mask
            # (X - cx)^2 + (Y - cy)^2 <= r^2
            dist_sq = (grid_x - cx)**2 + (grid_y - cy)**2
            drop_mask = dist_sq <= r2

            # Accumulate (Logical OR)
            # Unsqueeze to align with (N, 1, H, W)
            total_mask = total_mask | drop_mask.unsqueeze(1)

        # 7. Apply Mask
        # Where mask is True, use blurred image. Else use original.
        out = torch.where(total_mask, blurred_x, x)

        return out if is_batch else out.squeeze(0)


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # Create sharp test image (grid pattern) to see blur clearly
    img = torch.zeros(3, 256, 256)
    img[:, ::20, :] = 1.0 # Horizontal lines
    img[:, :, ::20] = 1.0 # Vertical lines
    
    # Augment
    aug = Waterdrops(
        probability=1.0, 
        amount=15, 
        radius_max=40,
        kernel_size=(25, 25),
        sigma=(10.0, 10.0) # Strong blur
    )
    
    res = aug(img)

    # Visualization
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    
    axs[0].imshow(img.permute(1, 2, 0))
    axs[0].set_title("Original (Grid)")
    axs[0].axis('off')
    
    axs[1].imshow(res.permute(1, 2, 0))
    axs[1].set_title("Waterdrops (Blurred Regions)")
    axs[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    print("Waterdrops test done.")