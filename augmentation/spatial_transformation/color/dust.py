# -*- coding: utf-8 -*-
"""
Dust/Artifact Augmentation.

Simulates dust particles, dead pixels, or sensor artifacts by setting
circular regions of the image to zero (black).

@author: Diyar Altinses, M.Sc.
"""

import torch


class Dust(torch.nn.Module):
    """
    Adds random circular "dust" artifacts (zeros) to the image.
    Efficiently computes a single mask accumulation before applying.
    """

    def __init__(
        self, 
        probability: float = 0.5, 
        amount: int = 100, 
        radius: float = 0.01,
        color: float = 0.0
    ):
        """
        Parameters
        ----------
        probability : float
            Probability of applying the augmentation (0.0 to 1.0).
        amount : int
            Number of dust particles to generate.
        radius : float
            Maximum radius of the particles relative to image diagonal.
            Example: 0.01 = 1% of image size.
        color : float
            Value to fill the dust holes with. 0.0 = Black, 1.0 = White.
        """
        super().__init__()
        self.probability = probability
        self.amount = amount
        self.radius = radius
        self.color = color

    def extra_repr(self) -> str:
        return f"probability={self.probability}, amount={self.amount}, radius={self.radius}, fill_color={self.color}"

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): Input image (C, H, W) or (N, C, H, W).
            
        Returns:
            torch.Tensor: Image with dust artifacts.
        """
        # 1. Probability Check
        if torch.rand(1, device=img.device) > self.probability:
            return img

        # 2. Setup Dimensions
        # Standardize input to (N, C, H, W)
        is_batch = img.ndim == 4
        x = img if is_batch else img.unsqueeze(0)
        
        N, C, H, W = x.shape
        device = x.device

        # 3. Create Coordinate Grid (once per batch)
        # Shape: (H, W)
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )

        # 4. Generate Random Parameters for ALL particles at once
        # Centers: (N, amount)
        center_y = torch.randint(0, H, (N, self.amount), device=device).float()
        center_x = torch.randint(0, W, (N, self.amount), device=device).float()
        
        # Radii: (N, amount)
        # Max radius in pixels
        img_diag = (H + W) / 2
        max_r_pix = int(img_diag * self.radius)
        if max_r_pix < 1: max_r_pix = 1
        
        radii = torch.randint(1, max_r_pix + 1, (N, self.amount), device=device).float()
        radii_sq = radii ** 2  # Compare squared distances to avoid sqrt() (faster)

        # 5. Accumulate Mask
        # We start with a mask of zeros (no dust)
        # Shape: (N, 1, H, W) to broadcast over channels
        total_mask = torch.zeros((N, 1, H, W), device=device, dtype=torch.bool)

        # Loop over the amount of particles.
        # Note: Fully vectorizing this (N * amount * H * W) would explode memory.
        # Iterating 'amount' times is a necessary tradeoff, but we keep operations lightweight.
        for i in range(self.amount):
            # Extract parameters for the i-th particle across all images in batch
            cy = center_y[:, i].view(N, 1, 1) # Shape (N, 1, 1) for broadcasting
            cx = center_x[:, i].view(N, 1, 1)
            r2 = radii_sq[:, i].view(N, 1, 1)

            # Calculate distance squared from current centers
            # (Y - cy)^2 + (X - cx)^2 < r^2
            dist_sq = (grid_y - cy) ** 2 + (grid_x - cx) ** 2
            
            # Create mask for this particle
            particle_mask = dist_sq <= r2
            
            # Logically OR with total mask (accumulate dust)
            total_mask = total_mask | particle_mask.unsqueeze(1)

        # 6. Apply Mask
        # Where mask is True, fill with self.color. Elsewhere keep original image.
        # We perform the copy to avoid modifying the input tensor in-place
        out = x.clone()
        color_tensor = torch.tensor(self.color, device=device, dtype=out.dtype).view(-1, 1, 1)
        out = torch.where(total_mask, color_tensor, out)

        return out if is_batch else out.squeeze(0)


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # Setup: 2 Images (Batch), White Background to see black dust
    batch_size = 2
    img_size = 256
    dummy_batch = torch.ones(batch_size, 3, img_size, img_size)

    # Init Augmentation
    # amount=50 particles, radius=2% of image size
    aug = Dust(probability=1.0, amount=50, radius=0.02, color=0.0)

    # Process
    result = aug(dummy_batch)

    # Visualization
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    
    axs[0].imshow(dummy_batch[0].permute(1, 2, 0))
    axs[0].set_title("Original (Clean)")
    axs[0].axis('off')

    axs[1].imshow(result[0].permute(1, 2, 0))
    axs[1].set_title("Augmented (Dust)")
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()
    print("Dust test completed.")