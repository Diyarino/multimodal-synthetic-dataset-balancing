# -*- coding: utf-8 -*-
"""
Scratches Augmentation.

Adds synthetic scratches/cracks to images using vectorized line drawing.
Replaces slow Python-loop Bresenham with fast PyTorch interpolation.

@author: Diyar Altinses, M.Sc.
"""

from typing import Optional
import torch


class Scratches(torch.nn.Module):
    """
    Simulates scratches or cracks on the image surface.
    """

    def __init__(
        self, 
        probability: float = 0.5, 
        num_scratches: int = 20, 
        max_length: int = 50, 
        max_width: int = 2,
        color: Optional[float] = None
    ):
        """
        Parameters
        ----------
        probability : float
            Probability of applying the augmentation.
        num_scratches : int
            Maximum number of scratches per image.
        max_length : int
            Maximum length of a scratch in pixels.
        max_width : int
            Maximum thickness of the scratch.
        color : float, optional
            The color/value of the scratch (e.g., 1.0 for white, 0.0 for black).
            If None, uses the image's max value (usually white).
        """
        super().__init__()
        self.probability = probability
        self.num_scratches = num_scratches
        self.max_length = max_length
        self.max_width = max_width
        self.color = color

    def extra_repr(self) -> str:
        return f"prob={self.probability}, num={self.num_scratches}, len={self.max_length}, width={self.max_width}"

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): Input image (C, H, W) or (N, C, H, W).
            
        Returns:
            torch.Tensor: Image with scratches.
        """
        # 1. Probability Check
        if torch.rand(1, device=img.device) > self.probability:
            return img

        # 2. Setup Input
        is_batch = img.ndim == 4
        x = img if is_batch else img.unsqueeze(0)
        N, C, H, W = x.shape
        device = x.device

        # Determine scratch color
        if self.color is not None:
            scratch_val = self.color
        else:
            # Auto-detect white (max value)
            scratch_val = x.max().item()

        # Output tensor (clone to avoid modifying input in-place)
        out = x.clone()

        # 3. Generate Random Scratches (Vectorized)
        
        # We generate 'num_scratches' for EACH image in the batch
        # Start points (N, num_scratches)
        x_start = torch.randint(0, W, (N, self.num_scratches), device=device)
        y_start = torch.randint(0, H, (N, self.num_scratches), device=device)
        
        # Lengths and Angles
        lengths = torch.randint(1, self.max_length + 1, (N, self.num_scratches), device=device).float()
        angles = torch.rand((N, self.num_scratches), device=device) * 2 * 3.14159 # 0 to 2pi

        # End points calculation (Polar to Cartesian)
        x_end = x_start.float() + lengths * torch.cos(angles)
        y_end = y_start.float() + lengths * torch.sin(angles)

        # 4. Draw Lines using Linear Interpolation (Efficient "Bresenham")
        # Instead of a while loop, we create points along the line
        
        # Max steps needed is the max length defined
        steps = int(self.max_length * 1.5) # 1.5x density ensures no holes in line
        t = torch.linspace(0, 1, steps, device=device).view(1, 1, steps) # (1, 1, Steps)
        
        # Expand start/end for broadcasting: (N, num_scratches, 1)
        xs = x_start.float().unsqueeze(-1)
        ys = y_start.float().unsqueeze(-1)
        xe = x_end.unsqueeze(-1)
        ye = y_end.unsqueeze(-1)
        
        # Interpolate points: P = P_start * (1-t) + P_end * t
        # Shape: (N, Num_Scratches, Steps)
        # This generates ALL coordinates for ALL lines in parallel!
        x_points = (xs * (1 - t) + xe * t).long()
        y_points = (ys * (1 - t) + ye * t).long()

        # 5. Clamp to Image Boundaries (Safety)
        x_points = x_points.clamp(0, W - 1)
        y_points = y_points.clamp(0, H - 1)

        # 6. Apply Thickness (Simulate width)
        # We offset the points slightly to create thickness
        # For simplicity, we just draw the points. 
        # (Handling width > 1 vectorially is complex, usually simple noise lines are thin)
        
        # Flatten batch and scratch dims to use indexing
        # We iterate per image in batch to apply indices
        for i in range(N):
            # Get all x, y coordinates for this image
            # Shape (Num_Scratches, Steps) -> Flatten to (Num_Scratches * Steps)
            xi = x_points[i].flatten()
            yi = y_points[i].flatten()
            
            # Draw
            # If width > 1, we can add neighbors (simple dilation)
            out[i, :, yi, xi] = scratch_val
            
            if self.max_width > 1:
                # Add simple thickness by drawing neighbors
                out[i, :, (yi + 1).clamp(0, H-1), xi] = scratch_val
                out[i, :, yi, (xi + 1).clamp(0, W-1)] = scratch_val

        return out if is_batch else out.squeeze(0)


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # Create black canvas
    img = torch.zeros(1, 3, 256, 256)
    
    # Augment
    aug = Scratches(
        probability=1.0, 
        num_scratches=50, 
        max_length=100, 
        max_width=2, 
        color=1.0
    )
    
    res = aug(img)

    # Visualization
    

    plt.figure(figsize=(6, 6))
    plt.imshow(res[0].permute(1, 2, 0))
    plt.title("Generated Scratches (Vectorized)")
    plt.axis('off')
    plt.show()
    print("Scratches test done.")