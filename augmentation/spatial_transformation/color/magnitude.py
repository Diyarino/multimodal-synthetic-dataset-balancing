# -*- coding: utf-8 -*-
"""
Image Inversion Augmentation.

Creates a negative of the image (Solarization/Inversion).
Works for both float (0-1) and byte (0-255) images.

@author: Diyar Altinses, M.Sc.
"""

import torch


class RandomInvert(torch.nn.Module):
    """
    Randomly inverts the colors of the input image (creates a negative).
    
    Formula:
        For Float images (0.0 - 1.0): out = 1.0 - input
        For Byte images (0 - 255):    out = 255 - input
    """

    def __init__(self, probability: float = 0.5):
        """
        Parameters
        ----------
        probability : float
            Probability of inverting the image.
        """
        super().__init__()
        self.probability = probability

    def extra_repr(self) -> str:
        return f"probability={self.probability}"

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): Input image. Shape (C, H, W) or (N, C, H, W).
            
        Returns:
            torch.Tensor: Inverted image.
        """
        # 1. Probability Check (Global per call, or per image)
        # Here we decide for the whole batch at once to keep it fast, 
        # or we can do per-image masking. Let's do per-image for better training variance.
        
        # If probability is 0, skip immediately
        if self.probability == 0:
            return img

        device = img.device
        
        # 2. Handle Dimensions (Batch vs Single)
        is_batch = img.ndim == 4
        x = img if is_batch else img.unsqueeze(0)
        N, C, H, W = x.shape

        # 3. Generate Mask for Batch
        # Shape (N, 1, 1, 1) so we invert specific images in the batch, but all channels/pixels of that image
        random_vals = torch.rand((N, 1, 1, 1), device=device)
        apply_mask = random_vals < self.probability
        
        # If no image in the batch needs inversion, return early
        if not apply_mask.any():
            return img

        # 4. Determine Range (0-1 or 0-255)
        # We assume floating point inputs are 0-1, unless max > 1
        is_float = x.is_floating_point()
        
        if is_float and x.max() <= 1.0:
            max_val = 1.0
        elif not is_float:
            # If uint8/byte
            max_val = 255
        else:
            # If float but values > 1 (e.g. unnormalized data)
            max_val = 255.0

        # 5. Apply Inversion
        # Logic: out = max_val - x
        # We only apply this where mask is True.
        
        # Create the inverted version
        inverted_x = max_val - x
        
        # Select: Either the inverted version or the original based on mask
        out = torch.where(apply_mask, inverted_x, x)

        return out if is_batch else out.squeeze(0)


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    # Create a dummy gradient image to see the effect clearly
    # Creates a gradient from 0 to 1
    gradient = np.linspace(0, 1, 256)
    gradient = np.tile(gradient, (256, 1))
    # Make it RGB
    img_np = np.stack([gradient, gradient, gradient], axis=2)
    
    # Convert to Tensor (Format: C, H, W, Value: 0-1)
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
    
    # Create Batch of 2: First one normal, second one inverted
    batch = img_tensor.unsqueeze(0).repeat(2, 1, 1, 1)

    # Init Augmentation (Probability 0.5 means roughly half the batch will be inverted)
    # We force prob=1.0 here to test the math, but in training you use 0.5
    aug = RandomInvert(probability=1.0)
    
    res = aug(batch)

    # Visualization
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    
    axs[0].imshow(img_tensor.permute(1, 2, 0).numpy())
    axs[0].set_title("Original (Gradient)")
    axs[0].axis('off')
    
    axs[1].imshow(res[0].permute(1, 2, 0).numpy())
    axs[1].set_title("Inverted (Negative)")
    axs[1].axis('off')
    
    plt.tight_layout()
    plt.show()