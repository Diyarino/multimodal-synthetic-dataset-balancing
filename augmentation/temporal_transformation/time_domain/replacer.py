# -*- coding: utf-8 -*-
"""
Random Erase / Cutout Augmentation.

Randomly replaces a rectangular region of the input with a constant value.
- For Time Series: Erases a segment of time across some channels.
- For Images: Erases a rectangular patch of pixels.

@author: Diyar Altinses, M.Sc.
"""

from typing import Tuple
import torch


class RandomErase(torch.nn.Module):
    """
    Randomly erases a rectangular region in the input tensor.
    Useful for forcing the model to learn distributed features (robustness to occlusion).
    """

    def __init__(
        self, 
        probability: float = 0.5, 
        scale_range: Tuple[float, float] = (0.02, 0.33), 
        ratio_range: Tuple[float, float] = (0.3, 3.3), 
        value: float = 0.0
    ):
        """
        Parameters
        ----------
        probability : float
            Probability of performing the operation.
        scale_range : tuple
            Range of proportion of total area to erase. (e.g., 0.02 to 0.33 of the image).
        ratio_range : tuple
            Range of aspect ratio of the erased region.
        value : float
            Value to fill the erased region with (default 0.0).
        """
        super().__init__()
        self.probability = probability
        self.scale_range = scale_range
        self.ratio_range = ratio_range
        self.value = value

    def extra_repr(self) -> str:
        return f"prob={self.probability}, scale={self.scale_range}, ratio={self.ratio_range}, val={self.value}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor. Shape (N, C, L) or (N, C, H, W).
            
        Returns:
            torch.Tensor: Augmented tensor.
        """
        # 1. Global Probability Check
        if torch.rand(1, device=x.device) > self.probability:
            return x

        # 2. Dimensions
        if x.ndim == 3: # (N, C, L) -> Treat as image (N, C, 1, L) or similar logic
            is_1d = True
            H, W = 1, x.shape[-1]
        else: # (N, C, H, W)
            is_1d = False
            H, W = x.shape[-2], x.shape[-1]
            
        N = x.shape[0]
        device = x.device

        # 3. Generate Random Box Parameters per Sample
        # Calculate random area and aspect ratio
        area = H * W
        target_areas = torch.empty(N, device=device).uniform_(*self.scale_range) * area
        aspect_ratios = torch.empty(N, device=device).uniform_(*self.ratio_range)
        
        # Calculate h and w for the box
        # h * w = target_area
        # w / h = aspect_ratio  => w = h * ratio
        # h * (h * ratio) = area => h = sqrt(area / ratio)
        
        h_cut = torch.sqrt(target_areas / aspect_ratios)
        w_cut = h_cut * aspect_ratios
        
        # Clamp sizes
        if is_1d:
            h_cut = torch.ones_like(h_cut) # Height is always 1 for 1D view
        else:
            h_cut = torch.clamp(h_cut, min=1, max=H)
            
        w_cut = torch.clamp(w_cut, min=1, max=W)
        
        # Cast to int
        h_cut = h_cut.long()
        w_cut = w_cut.long()
        
        # Random positions
        # Top-left corner (y, x)
        # y must be in [0, H - h_cut]
        # x must be in [0, W - w_cut]
        
        rand_y = torch.rand(N, device=device)
        rand_x = torch.rand(N, device=device)
        
        y1 = (rand_y * (H - h_cut)).long()
        x1 = (rand_x * (W - w_cut)).long()
        
        # 4. Create Mask (Vectorized)
        # We construct a boolean mask for the holes
        
        # Grid Y: (1, H, 1)
        grid_y = torch.arange(H, device=device).view(1, H, 1)
        # Grid X: (1, 1, W)
        grid_x = torch.arange(W, device=device).view(1, 1, W)
        
        # Sample parameters reshaped for broadcasting: (N, 1, 1)
        y1 = y1.view(N, 1, 1)
        x1 = x1.view(N, 1, 1)
        h_cut = h_cut.view(N, 1, 1)
        w_cut = w_cut.view(N, 1, 1)
        
        # Mask Y: True where y is inside the box range
        mask_y = (grid_y >= y1) & (grid_y < y1 + h_cut)
        # Mask X: True where x is inside the box range
        mask_x = (grid_x >= x1) & (grid_x < x1 + w_cut)
        
        # Final Mask: (N, H, W) -> Intersection of Y and X masks
        mask = mask_y & mask_x
        
        # Expand channel dim: (N, 1, H, W)
        mask = mask.unsqueeze(1)
        
        # 5. Apply Cutout
        if is_1d:
             # Mask is (N, 1, 1, L), x is (N, C, L) -> broadcast works if we remove H dim
             mask = mask.squeeze(2) # (N, 1, L)
             
        out = x.clone()
        out.masked_fill_(mask, self.value)

        return out


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # ---------------------------------------------------------
    # Test 1: 2D Image
    # ---------------------------------------------------------
    H, W = 100, 100
    img = torch.ones(3, 1, H, W) # 3 White Images
    
    # Augment: Erase a rectangle
    aug_2d = RandomErase(probability=1.0, value=0.0)
    out_2d = aug_2d(img)
    
    # ---------------------------------------------------------
    # Test 2: 1D Signal
    # ---------------------------------------------------------
    L = 200
    signal = torch.ones(3, 1, L) # 3 Flat Lines
    
    # Augment: Erase a segment
    aug_1d = RandomErase(probability=1.0, value=0.0)
    out_1d = aug_1d(signal)

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------
    

    fig, axs = plt.subplots(2, 3, figsize=(10, 6))
    
    # Row 1: Images
    for i in range(3):
        axs[0, i].imshow(out_2d[i, 0], cmap='gray', vmin=0, vmax=1)
        axs[0, i].set_title(f"Image Sample {i}")
        axs[0, i].axis('off')

    # Row 2: Signals
    for i in range(3):
        axs[1, i].plot(out_1d[i, 0].numpy())
        axs[1, i].set_ylim(-0.1, 1.1)
        axs[1, i].set_title(f"Signal Sample {i}")
        
    plt.tight_layout()
    plt.show()
    print("RandomErase test done.")