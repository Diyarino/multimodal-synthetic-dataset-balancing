# -*- coding: utf-8 -*-
"""
Zero-One Normalization Augmentation.

Scales input tensor values to the range [0, 1].
This is a standard preprocessing step for many neural networks (e.g., Sigmoid outputs).

@author: Diyar Altinses, M.Sc.
"""

import torch


class NormalizeZeroOne(torch.nn.Module):
    """
    Normalizes the input tensor to the range [0, 1].
    Formula: X_out = (X - X_min) / (X_max - X_min + epsilon)
    
    Robust against constant images (division by zero protection) and handles batches correctly.
    """

    def __init__(self, probability: float = 1.0, epsilon: float = 1e-8):
        """
        Parameters
        ----------
        probability : float
            Probability of applying the normalization.
        epsilon : float
            Small value to avoid division by zero if image is constant (max == min).
        """
        super().__init__()
        self.probability = probability
        self.epsilon = epsilon

    def extra_repr(self) -> str:
        return f"probability={self.probability}, epsilon={self.epsilon}"

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): Input tensor. Shape (C, H, W) or (N, C, H, W).
            
        Returns:
            torch.Tensor: Normalized tensor in range [0, 1].
        """
        # 1. Probability Check (on correct device)
        if torch.rand(1, device=img.device) > self.probability:
            return img

        # 2. Prepare Input (Must be float for division)
        x = img.float() if not img.is_floating_point() else img

        # 3. Calculate Min/Max (Per Sample logic)
        if x.ndim == 4:
            # Batch mode: (N, C, H, W) -> Min/Max per image N
            # Flatten spatial dims: (N, C*H*W)
            x_flat = x.view(x.size(0), -1)
            
            # Get values (N, 1)
            batch_min = x_flat.min(dim=1, keepdim=True)[0]
            batch_max = x_flat.max(dim=1, keepdim=True)[0]
            
            # Reshape for broadcasting: (N, 1, 1, 1)
            current_min = batch_min.view(-1, 1, 1, 1)
            current_max = batch_max.view(-1, 1, 1, 1)
        else:
            # Single image mode
            current_min = x.min()
            current_max = x.max()

        # 4. Apply Normalization
        numerator = x - current_min
        denominator = (current_max - current_min) + self.epsilon
        
        out = numerator / denominator

        # 5. Restore Type (Optional)
        # Usually we keep normalized data as float. 
        # Only cast back if original was not float AND we explicitly want to (rare for 0-1 norm).
        # We return float here as standard behavior for [0, 1] tensors.
        
        return out


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # Create a batch of 2 dummy images
    # Image 1: Range 10-20 (Low contrast)
    # Image 2: Range 0-255 (High contrast)
    img = torch.zeros(2, 1, 100, 100)
    img[0] = torch.linspace(10, 20, 100).unsqueeze(0).expand(100, 100)
    img[1] = torch.linspace(0, 255, 100).unsqueeze(0).expand(100, 100)
    
    print(f"Original Ranges: Img1 [{img[0].min():.1f}, {img[0].max():.1f}], Img2 [{img[1].min():.1f}, {img[1].max():.1f}]")

    # Normalize
    aug = NormalizeZeroOne(probability=1.0)
    out = aug(img)
    
    print(f"Norm Ranges:     Img1 [{out[0].min():.1f}, {out[0].max():.1f}], Img2 [{out[1].min():.1f}, {out[1].max():.1f}]")

    # Visualization
    
    
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    
    axs[0].imshow(img[0, 0], cmap='gray', vmin=0, vmax=255)
    axs[0].set_title("Original (Values 10-20)\nLooks Dark/Gray")
    axs[0].axis('off')
    
    axs[1].imshow(out[0, 0], cmap='gray', vmin=0, vmax=1)
    axs[1].set_title("Normalized [0, 1]\nContrast Stretched (Looks Full Range)")
    axs[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    print("NormalizeZeroOne test done.")