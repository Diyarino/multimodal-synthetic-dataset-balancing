# -*- coding: utf-8 -*-
"""
Warping Augmentations (Time & Magnitude).

Uses smooth deformations (Splines / Interpolation) to distort the signal.
1. TimeWarping: Elastic deformation of the time axis (Speed up / Slow down locally).
2. MagnitudeWarping: Multiplies signal by a smooth curve (Volume changes).

@author: Diyar Altinses, M.Sc.
"""

import torch
import torch.nn as nn


class TimeWarping(torch.nn.Module):
    """
    Applies elastic deformation to the time axis.
    
    Instead of inserting/deleting points (which breaks batches), 
    we use a smooth flow field to resample the signal.
    This simulates local variations in speed (e.g. gesture performed faster then slower).
    """

    def __init__(self, probability: float = 1.0, warp_steps: int = 5, strength: float = 0.5):
        """
        Parameters
        ----------
        probability : float
            Probability of applying the augmentation.
        warp_steps : int
            Number of control points (knots) for the warping curve.
            Higher values = more rapid speed changes (more wiggly time).
            Lower values = smoother speed changes.
        strength : float
            Max distortion strength. 
            0.5 means a point can be shifted by up to half the signal length (extreme!).
            Typical values: 0.1 to 0.3.
        """
        super().__init__()
        self.probability = probability
        self.warp_steps = warp_steps
        self.strength = strength

    def extra_repr(self) -> str:
        return f"probability={self.probability}, knots={self.warp_steps}, strength={self.strength}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor (N, C, L).
            
        Returns:
            torch.Tensor: Warped tensor (N, C, L).
        """
        if torch.rand(1, device=x.device) > self.probability:
            return x

        N, C, L = x.shape
        device = x.device

        # 1. Create the Flow Field
        # We generate random noise at low resolution (warp_steps)
        # and upsample it to length L. This creates a smooth curve.
        
        # Random offsets: (N, 1, 1, warp_steps)
        # We treat this as a 1-pixel high image for grid_sample compatibility
        noise_low = torch.randn(N, 1, 1, self.warp_steps, device=device) * self.strength
        
        # Upsample to signal length L using bicubic or linear interpolation
        # Result: Smooth random curve
        noise_smooth = torch.nn.functional.interpolate(
            noise_low, 
            size=(1, L), 
            mode='bicubic', 
            align_corners=True
        ) # Shape (N, 1, 1, L)
        
        # Squeeze to (N, 1, L)
        flow = noise_smooth.squeeze(2) 

        # 2. Create the Grid
        # Standard grid from -1 to 1
        # Shape (N, L)
        orig_grid = torch.linspace(-1, 1, L, device=device).view(1, 1, L).expand(N, -1, -1)
        
        # Add flow to grid -> Distorted time
        warped_grid_x = orig_grid + flow
        
        # Clip to ensure we stay within signal bounds [-1, 1]
        warped_grid_x = torch.clamp(warped_grid_x, -1, 1)

        # 3. Apply Grid Sample
        # grid_sample expects (N, H, W, 2) for (x, y) coordinates.
        # Since we are 1D, we use H=1. Y-coordinate is always 0.
        
        # Prepare coordinates: (N, 1, L, 2)
        # X = warped_grid_x, Y = 0
        grid_y = torch.zeros_like(warped_grid_x)
        grid = torch.stack((warped_grid_x, grid_y), dim=-1).permute(0, 2, 3, 1) # (N, 1, L, 2) -> Wait, stack dims...
        
        # Correct stack shape:
        # warped_grid_x is (N, 1, L)
        # We need (N, 1, L, 2)
        grid = torch.cat([warped_grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], dim=-1)

        # Input needs to be 4D for grid_sample: (N, C, 1, L)
        x_4d = x.unsqueeze(2)
        
        # Sample!
        # align_corners=True matches the grid definition -1 to 1
        warped_x = torch.nn.functional.grid_sample(x_4d, grid, mode='bilinear', padding_mode='border', align_corners=True)
        
        # Remove dummy dimension
        return warped_x.squeeze(2)


class MagnitudeWarping(nn.Module):
    """
    Multiplies the signal by a smooth random curve (Spline).
    Simulates amplitude drift or changing sensor sensitivity.
    """

    def __init__(self, probability: float = 1.0, knots: int = 4, sigma: float = 0.2):
        """
        Parameters
        ----------
        probability : float
            Probability.
        knots : int
            Number of random points to interpolate between.
            4 knots = The curve changes direction roughly 4 times.
        sigma : float
            Standard deviation of the curve.
            0.2 means the curve fluctuates roughly around 1.0 +/- 0.2.
        """
        super().__init__()
        self.probability = probability
        self.knots = knots
        self.sigma = sigma

    def extra_repr(self) -> str:
        return f"probability={self.probability}, knots={self.knots}, sigma={self.sigma}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor (N, C, L).
            
        Returns:
            torch.Tensor: Tensor multiplied by smooth curve.
        """
        if torch.rand(1, device=x.device) > self.probability:
            return x

        N, C, L = x.shape
        device = x.device

        # 1. Generate Random Knots
        # Shape: (N, C, knots) -> One curve per channel!
        # Values around 1.0 (Normal distribution + 1)
        random_knots = torch.randn(N, C, self.knots, device=device) * self.sigma + 1.0
        
        # 2. Smooth Interpolation (Upsampling)
        # Use F.interpolate to generate the smooth curve from knots
        # interpolation expects (Batch, Channels, Length)
        
        # Linear mode is simplest smooth path. Bicubic is smoother but requires 4D input generally.
        # For 1D, linear or cubic via 1D conv is fine. F.interpolate 'linear' works for 3D tensors.
        smooth_curve = torch.nn.functional.interpolate(
            random_knots, 
            size=L, 
            mode='linear', 
            align_corners=True
        )
        
        # 3. Apply
        return x * smooth_curve


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # ---------------------------------------------------------
    # Test Setup: Sine Wave
    # ---------------------------------------------------------
    t = torch.linspace(0, 4 * 3.14, 200)
    signal = torch.sin(t).view(1, 1, 200)
    
    # ---------------------------------------------------------
    # 1. Time Warping
    # ---------------------------------------------------------
    # knots=5 means "wiggle time" 5 times across the signal
    aug_time = TimeWarping(probability=1.0, warp_steps=5, strength=0.3)
    out_time = aug_time(signal.clone())

    # ---------------------------------------------------------
    # 2. Magnitude Warping
    # ---------------------------------------------------------
    aug_mag = MagnitudeWarping(probability=1.0, knots=5, sigma=0.5)
    out_mag = aug_mag(signal.clone())
    
    # Capture the curve itself for visualization (hack)
    curve = out_mag / (signal + 1e-6) 

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------
    
    

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot Time Warping
    axs[0].plot(signal[0, 0].numpy(), 'k--', alpha=0.5, label="Original")
    axs[0].plot(out_time[0, 0].numpy(), 'b', label="Time Warped")
    axs[0].set_title("Time Warping (Elastic Deformation)\nNote: Peaks shift left/right")
    axs[0].legend()
    
    # Plot Magnitude Warping
    axs[1].plot(signal[0, 0].numpy(), 'k--', alpha=0.5, label="Original")
    axs[1].plot(out_mag[0, 0].numpy(), 'r', label="Mag Warped")
    # Plot the envelope (optional, just to show what happened)
    # We plot it scaled down or separate usually, here just overlay
    axs[1].set_title("Magnitude Warping (Smooth Envelope)\nNote: Amplitude swells and fades")
    axs[1].legend()
    
    plt.tight_layout()
    plt.show()
    print("Warping Augmentations test done.")