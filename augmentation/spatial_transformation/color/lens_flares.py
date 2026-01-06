# -*- coding: utf-8 -*-
"""
Lens Flare / Dirt Augmentation.

Simulates dirty lenses or light flares by adding blurred, colored geometric shapes.
Optimized for PyTorch (GPU/Batch support).

@author: Diyar Altinses, M.Sc.
"""

from typing import Tuple
import torch
from torchvision.transforms import GaussianBlur

class Flares(torch.nn.Module):
    """
    Adds blurred "flare" or "dirt" artifacts to the image.
    Supports circular and rectangular shapes with additive color blending.
    """

    def __init__(
        self, 
        probability: float = 0.5, 
        num_flares: int = 10, 
        radius_range: Tuple[int, int] = (5, 20),
        mode: str = 'circ', 
        kernel_size: Tuple[int, int] = (25, 25), 
        sigma: Tuple[float, float] = (15.0, 20.0),
        opacity: float = 0.5
    ):
        """
        Parameters
        ----------
        probability : float
            Probability of applying the transform.
        num_flares : int
            Maximum number of flares to generate per image.
        radius_range : tuple
            (min, max) radius of the flares in pixels.
        mode : str
            'circ' (circles) or 'rect' (squares).
        kernel_size : tuple
            Size of the Gaussian blur kernel. Must be odd numbers.
        sigma : tuple
            (min, max) standard deviation for the blur.
        opacity : float
            Strength of the flare overlay (0.0 to 1.0).
        """
        super().__init__()
        self.probability = probability
        self.num_flares = num_flares
        self.radius_min, self.radius_max = radius_range
        self.mode = mode
        self.opacity = opacity
        
        # GaussianBlur expects kernel size to be odd
        ks = list(kernel_size)
        if ks[0] % 2 == 0: ks[0] += 1
        if ks[1] % 2 == 0: ks[1] += 1
        self.blur = GaussianBlur(kernel_size=tuple(ks), sigma=sigma)

    def extra_repr(self) -> str:
        return f"prob={self.probability}, num={self.num_flares}, radius={self.radius_min}-{self.radius_max}, mode={self.mode}"

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): Input image (C, H, W) or (N, C, H, W).
            
        Returns:
            torch.Tensor: Image with added flares.
        """
        # 1. Probability Check
        if torch.rand(1, device=img.device) > self.probability:
            return img

        # 2. Setup Inputs
        is_batch = img.ndim == 4
        x = img if is_batch else img.unsqueeze(0)
        
        N, C, H, W = x.shape
        device = x.device
        
        # Detect range (0-1 float or 0-255 byte)
        is_float = x.is_floating_point()
        max_val = 1.0 if is_float and x.max() <= 1.0 else 255.0
        
        # Work in Float for blending
        x_float = x.float()

        # 3. Generate Parameters (Batch-wise)
        # Random count of flares per image (1 to num_flares)
        # To simplify vectorization, we generate 'num_flares' for everyone 
        # but zero out some based on a random mask, or just use fixed num_flares.
        # Here we use fixed num_flares for max performance, or you can loop.
        
        # Centers: (N, num_flares)
        cx = torch.randint(0, W, (N, self.num_flares), device=device).float()
        cy = torch.randint(0, H, (N, self.num_flares), device=device).float()
        
        # Radii: (N, num_flares)
        radii = torch.randint(self.radius_min, self.radius_max + 1, (N, self.num_flares), device=device).float()
        
        # Colors: (N, num_flares, C) -> Random color per flare
        colors = torch.rand(N, self.num_flares, C, device=device) * max_val

        # 4. Draw Flares (Vectorized Accumulation)
        # Canvas for flares: (N, C, H, W)
        flare_layer = torch.zeros_like(x_float)
        
        # Grid for mask creation: (H, W)
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32), 
            torch.arange(W, device=device, dtype=torch.float32), 
            indexing='ij'
        )
        
        

        for i in range(self.num_flares):
            # Extract i-th flare params for all images in batch
            # Shape (N, 1, 1) for broadcasting against (H, W)
            curr_cx = cx[:, i].view(N, 1, 1)
            curr_cy = cy[:, i].view(N, 1, 1)
            curr_r = radii[:, i].view(N, 1, 1)
            curr_color = colors[:, i].view(N, C, 1, 1) # (N, C, 1, 1)

            if self.mode == 'circ':
                # Circle Mask: (x-cx)^2 + (y-cy)^2 <= r^2
                # Result shape broadcast: (N, H, W)
                dist_sq = (grid_x - curr_cx)**2 + (grid_y - curr_cy)**2
                mask = dist_sq <= (curr_r**2)
            else:
                # Rect Mask: |x-cx| <= r AND |y-cy| <= r
                mask_x = (grid_x - curr_cx).abs() <= curr_r
                mask_y = (grid_y - curr_cy).abs() <= curr_r
                mask = mask_x & mask_y

            # Add Color to Layer
            # Mask is (N, H, W) -> unsqueeze to (N, 1, H, W)
            # curr_color is (N, C, 1, 1)
            # We additively blend flares (overlapping flares get brighter)
            mask = mask.unsqueeze(1)
            flare_layer += mask.float() * curr_color

        # 5. Blur the flare layer
        # Blur needs to happen on the accumulated layer to look like out-of-focus dirt
        flare_layer_blurred = self.blur(flare_layer)

        # 6. Blend with Original Image
        # Additive blending: Image + Opacity * Flare
        out = x_float + (self.opacity * flare_layer_blurred)
        
        # 7. Clamp and Restore Type
        out = out.clamp(0, max_val)
        
        if not is_float:
            out = out.to(dtype=img.dtype)
            
        return out if is_batch else out.squeeze(0)


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # Dummy Image (Black background to see flares clearly)
    # Shape: (1, 3, 256, 256)
    dummy_img = torch.zeros(1, 3, 256, 256)
    
    # Initialize
    aug = Flares(
        probability=1.0, 
        num_flares=15, 
        radius_range=(5, 30), 
        mode='circ', 
        opacity=0.8,
        kernel_size=(25, 25) # Large blur
    )
    
    # Forward
    res = aug(dummy_img)
    
    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(dummy_img[0].permute(1, 2, 0).int()) # .int() for 0-255 display
    axs[0].set_title("Original (Black)")
    
    # Normalize result for display if it exceeded 255 slightly or is float
    res_disp = res[0].permute(1, 2, 0)
    if res_disp.max() > 1.0:
        res_disp = res_disp / 255.0
        
    axs[1].imshow(res_disp)
    axs[1].set_title("Flares (Blurred)")
    
    plt.show()
    print("Flares test done.")