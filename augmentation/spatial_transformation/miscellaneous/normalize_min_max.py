# -*- coding: utf-8 -*-
"""
Min-Max Normalization Augmentation.

Rescales the pixel values of images to a specific range (e.g., [0, 1] or [-1, 1]).
Essential for preprocessing neural network inputs.

@author: Diyar Altinses, M.Sc.
"""

import torch


class NormalizeMinMax(torch.nn.Module):
    """
    Normalizes the input tensor to a specific range [min_val, max_val].
    Formula: X_norm = (X - X_min) / (X_max - X_min) * (target_max - target_min) + target_min
    
    Robust against constant images (division by zero protection).
    """

    def __init__(
        self, 
        probability: float = 1.0, 
        min_val: float = 0.0, 
        max_val: float = 1.0,
        epsilon: float = 1e-8
    ):
        """
        Parameters
        ----------
        probability : float
            Probability of applying the normalization.
        min_val : float
            Target minimum value (e.g. 0.0 or -1.0).
        max_val : float
            Target maximum value (e.g. 1.0).
        epsilon : float
            Small value to avoid division by zero if image is constant.
        """
        super().__init__()
        self.probability = probability
        self.min_val = min_val
        self.max_val = max_val
        self.epsilon = epsilon

    def extra_repr(self) -> str:
        return f"probability={self.probability}, target_range=[{self.min_val}, {self.max_val}]"

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): Input image. Shape (C, H, W) or (N, C, H, W).
            
        Returns:
            torch.Tensor: Normalized image.
        """
        # 1. Probability Check
        if torch.rand(1, device=img.device) > self.probability:
            return img

        # 2. Prepare Input
        # Normalization requires float division
        x = img.float() if not img.is_floating_point() else img
        
        # 3. Calculate Min/Max per Sample
        # We need to handle (C, H, W) vs (N, C, H, W) differently to normalize per image.
        
        if x.ndim == 4:
            # Batch mode: (N, C, H, W) -> Compute stats per image (N)
            # Flatten spatial dims to find min/max per batch item
            x_flat = x.view(x.size(0), -1)
            batch_min = x_flat.min(dim=1, keepdim=True)[0] # Shape (N, 1)
            batch_max = x_flat.max(dim=1, keepdim=True)[0] # Shape (N, 1)
            
            # Reshape for broadcasting: (N, 1, 1, 1)
            current_min = batch_min.view(-1, 1, 1, 1)
            current_max = batch_max.view(-1, 1, 1, 1)
            
        else:
            # Single image mode: (C, H, W) -> Global min/max
            current_min = x.min()
            current_max = x.max()

        # 4. Apply Normalization Formula
        # (X - min) / (max - min + eps)
        numerator = x - current_min
        denominator = (current_max - current_min) + self.epsilon
        
        x_norm = numerator / denominator
        
        # 5. Scale to Target Range
        # X_norm * (new_max - new_min) + new_min
        target_range = self.max_val - self.min_val
        out = x_norm * target_range + self.min_val

        # 6. Restore Type (Optional)
        # Usually, normalized images SHOULD remain float. 
        # Converting back to uint8 destroys the normalization if range is small (e.g. 0-1).
        # We only cast back if the input wasn't float AND the target range allows it (e.g. 0-255).
        
        if not img.is_floating_point() and self.max_val > 1.0:
            out = out.to(img.dtype)

        return out


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # Create a batch of 2 images with very different ranges
    # Image 1: Dark (values 0 to 50)
    # Image 2: Bright (values 200 to 255)
    img = torch.zeros(2, 1, 100, 100)
    
    # Fill Image 1 with a ramp 0->50
    img[0] = torch.linspace(0, 50, 100).unsqueeze(0).expand(100, 100)
    
    # Fill Image 2 with a ramp 200->255
    img[1] = torch.linspace(200, 255, 100).unsqueeze(0).expand(100, 100)
    
    print(f"Original Ranges: Img1 [{img[0].min():.1f}, {img[0].max():.1f}], Img2 [{img[1].min():.1f}, {img[1].max():.1f}]")

    # Normalize both to strictly [0, 1]
    aug = NormalizeMinMax(probability=1.0, min_val=0.0, max_val=1.0)
    out = aug(img)
    
    print(f"Norm Ranges:     Img1 [{out[0].min():.1f}, {out[0].max():.1f}], Img2 [{out[1].min():.1f}, {out[1].max():.1f}]")

    # Visualization
    

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    
    # Original
    axs[0, 0].imshow(img[0, 0], cmap='gray', vmin=0, vmax=255)
    axs[0, 0].set_title("Original Dark\nRange: 0-50")
    
    axs[0, 1].imshow(img[1, 0], cmap='gray', vmin=0, vmax=255)
    axs[0, 1].set_title("Original Bright\nRange: 200-255")
    
    # Normalized (Display vmin=0, vmax=1)
    axs[1, 0].imshow(out[0, 0], cmap='gray', vmin=0, vmax=1)
    axs[1, 0].set_title("Normalized\nRange: 0.0-1.0")
    
    axs[1, 1].imshow(out[1, 0], cmap='gray', vmin=0, vmax=1)
    axs[1, 1].set_title("Normalized\nRange: 0.0-1.0")
    
    plt.tight_layout()
    plt.show()