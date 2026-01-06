# -*- coding: utf-8 -*-
"""
Periodic Noise Augmentation.

Adds structured, periodic interference patterns (stripes/waves) to the image.
Commonly simulates electrical interference (e.g. 50Hz hum) or scanline artifacts.

@author: Diyar Altinses, M.Sc.
"""

from typing import Tuple
import torch
import math


class PeriodicNoise(torch.nn.Module):
    """
    Adds periodic noise (sine/cosine waves) to the input tensor.
    Can generate horizontal, vertical, or diagonal interference patterns.
    """

    def __init__(
        self, 
        probability: float = 0.5, 
        max_amplitude: float = 0.2, 
        max_frequency: int = 32
    ):
        """
        Parameters
        ----------
        probability : float
            Probability of applying the noise.
        max_amplitude : float
            Maximum strength of the stripes (relative to image max value).
        max_frequency : int
            Maximum number of stripe lines across the image.
        """
        super().__init__()
        self.probability = probability
        self.max_amplitude = max_amplitude
        self.max_frequency = max_frequency

    def extra_repr(self) -> str:
        return f"probability={self.probability}, max_amp={self.max_amplitude}, max_freq={self.max_frequency}"

    def _generate_grid(self, h: int, w: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Helper to create coordinate grids efficiently."""
        # indexing='ij' ensures:
        # Y varies along Height (dim 0) -> Horizontal lines
        # X varies along Width (dim 1) -> Vertical lines
        y = torch.linspace(0, 1, h, device=device)
        x = torch.linspace(0, 1, w, device=device)
        return torch.meshgrid(y, x, indexing='ij')

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): Input image (C, H, W) or (N, C, H, W).
            
        Returns:
            torch.Tensor: Image with periodic stripes.
        """
        # 1. Probability Check
        if torch.rand(1, device=img.device) > self.probability:
            return img

        # 2. Setup Dimensions
        is_batch = img.ndim == 4
        x = img if is_batch else img.unsqueeze(0)
        N, C, H, W = x.shape
        device = x.device

        # 3. Determine Parameters (Random per call)
        # Random frequency (number of lines)
        freq = torch.randint(1, self.max_frequency + 1, (1,), device=device).item()
        
        # Random amplitude (strength)
        amp = torch.rand(1, device=device).item() * self.max_amplitude
        
        # Random Mode: 0=Horizontal, 1=Vertical, 2=Diagonal/Complex
        mode = torch.randint(0, 3, (1,), device=device).item()

        # 4. Generate Wave Pattern
        Y, X = self._generate_grid(H, W, device)
        
        # Determine max value for scaling (255 or 1.0)
        max_val = 255.0 if (not x.is_floating_point() or x.max() > 1.0) else 1.0

        if mode == 0:  # Horizontal Stripes (varies along Y)
            # Pattern shape (H, W)
            noise = torch.cos(freq * math.pi * Y)
            
        elif mode == 1: # Vertical Stripes (varies along X)
            noise = torch.cos(freq * math.pi * X)
            
        else: # Multi / Diagonal
            # Combination of waves
            noise = torch.cos(freq * math.pi * Y) + torch.sin(freq * math.pi * (X + Y))
            # Normalize complex wave back to roughly [-1, 1] range
            noise = noise / 2.0

        # 5. Apply Noise
        # Expand noise to (1, 1, H, W) for broadcasting
        noise = noise.view(1, 1, H, W) * amp * max_val
        
        # Add to input
        out = x + noise

        # 6. Clamp and Restore
        if not x.is_floating_point():
            out = out.clamp(0, 255).to(x.dtype)
        else:
            out = out.clamp(0, max_val)

        return out if is_batch else out.squeeze(0)


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # Create a simple gray image
    H, W = 256, 256
    img = torch.full((1, 1, H, W), 0.5)
    
    # Init Augmentation (High amplitude to see it clearly)
    aug = PeriodicNoise(probability=1.0, max_amplitude=0.3, max_frequency=20)
    
    # Run multiple times to see different modes
    # We force the internal random generator to show different modes here visually?
    # Since we can't force the random mode easily without changing code, 
    # we just run it 3 times and hope for variety or rely on probability.
    
    out1 = aug(img)
    out2 = aug(img)
    out3 = aug(img)

    # Visualization
    

    fig, axs = plt.subplots(1, 4, figsize=(12, 3))
    
    axs[0].imshow(img[0, 0], cmap='gray', vmin=0, vmax=1)
    axs[0].set_title("Original (Flat Gray)")
    axs[0].axis('off')

    axs[1].imshow(out1[0, 0], cmap='gray', vmin=0, vmax=1)
    axs[1].set_title("Sample 1")
    axs[1].axis('off')

    axs[2].imshow(out2[0, 0], cmap='gray', vmin=0, vmax=1)
    axs[2].set_title("Sample 2")
    axs[2].axis('off')
    
    axs[3].imshow(out3[0, 0], cmap='gray', vmin=0, vmax=1)
    axs[3].set_title("Sample 3")
    axs[3].axis('off')
    
    plt.tight_layout()
    plt.show()
    print("PeriodicNoise test done.")