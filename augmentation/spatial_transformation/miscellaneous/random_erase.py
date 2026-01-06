# -*- coding: utf-8 -*-
"""
Random Erase / Cutout Augmentation.

Randomly selects a rectangular region in the image and erases it 
(fills with a constant value or noise). 
Proven to improve robustness (e.g., Cutout paper).

@author: Diyar Altinses, M.Sc.
"""

from typing import Tuple, Optional
import torch


class RandomErase(torch.nn.Module):
    """
    Randomly selects a rectangle region in an image and erases its pixels.
    'Cutout' regularization technique.
    """

    def __init__(
        self, 
        probability: float = 0.5, 
        size: Optional[Tuple[int, int]] = None, 
        scale: Tuple[float, float] = (0.02, 0.33),
        ratio: Tuple[float, float] = (0.3, 3.3),
        value: float = 0.0
    ):
        """
        Parameters
        ----------
        probability : float
            Probability of performing the operation.
        size : tuple, optional
            Fixed size (height, width) of the erased area. 
            If provided, 'scale' and 'ratio' are ignored.
        scale : tuple
            Range of proportion of erased area against input image (if size is None).
        ratio : tuple
            Range of aspect ratio of erased area (if size is None).
        value : float
            Pixel value to fill the erased area with (0 for black, 1 for white, etc.).
        """
        super().__init__()
        self.probability = probability
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def extra_repr(self) -> str:
        return f"prob={self.probability}, size={self.size}, scale={self.scale}, ratio={self.ratio}, val={self.value}"

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): Input image (C, H, W) or (N, C, H, W).
            
        Returns:
            torch.Tensor: Image with erased rectangle.
        """
        # 1. Probability Check
        if torch.rand(1, device=img.device) > self.probability:
            return img

        # 2. Setup Dimensions
        is_batch = img.ndim == 4
        x = img if is_batch else img.unsqueeze(0)
        N, C, H, W = x.shape
        
        # Clone to avoid in-place modification of original tensor
        out = x.clone()

        # 3. Apply Erasing per Image in Batch
        # Since generating variable sized rectangles for a whole batch at once 
        # is complex to vectorize without masking, a simple loop is often cleaner 
        # and fast enough for this specific augmentation.
        
        for i in range(N):
            # Calculate Rectangle Size (h_cut, w_cut)
            if self.size is not None:
                h_cut, w_cut = self.size
                # Clip to image size just in case
                h_cut = min(h_cut, H)
                w_cut = min(w_cut, W)
            else:
                # Randomly determine size based on scale and aspect ratio
                area = H * W
                target_area = torch.empty(1).uniform_(self.scale[0], self.scale[1]).item() * area
                aspect_ratio = torch.empty(1).uniform_(self.ratio[0], self.ratio[1]).item()

                w_cut = int(round((target_area * aspect_ratio) ** 0.5))
                h_cut = int(round((target_area / aspect_ratio) ** 0.5))

                # Safety checks
                if w_cut > W: w_cut = W
                if h_cut > H: h_cut = H
                if w_cut < 1: w_cut = 1
                if h_cut < 1: h_cut = 1

            # Calculate Position (Top-Left corner)
            y = torch.randint(0, H - h_cut + 1, (1,)).item()
            x_pos = torch.randint(0, W - w_cut + 1, (1,)).item()

            # Erase
            # out[i, :, y:y+h, x:x+w] = value
            out[i, :, y : y + h_cut, x_pos : x_pos + w_cut] = self.value

        return out if is_batch else out.squeeze(0)


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # Create a batch of 2 dummy images (Gradient + Noise)
    # Shape: (2, 3, 100, 100)
    H, W = 100, 100
    img = torch.rand(2, 3, H, W)
    
    # Augment: Fixed size box vs Random scale box
    
    # 1. Fixed Size (e.g. 30x30)
    aug_fixed = RandomErase(probability=1.0, size=(30, 30), value=0.0)
    res_fixed = aug_fixed(img)

    # 2. Random Scale (Standard Cutout) - Fills with Gray (0.5)
    aug_random = RandomErase(probability=1.0, size=None, scale=(0.1, 0.4), value=0.5)
    res_random = aug_random(img)

    # Visualization
    

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    
    # Fixed Size Results
    axs[0, 0].imshow(res_fixed[0].permute(1, 2, 0))
    axs[0, 0].set_title("Fixed Size (30x30) - Img 1")
    axs[0, 1].imshow(res_fixed[1].permute(1, 2, 0))
    axs[0, 1].set_title("Fixed Size (30x30) - Img 2\n(Different Position!)")
    
    # Random Scale Results
    axs[1, 0].imshow(res_random[0].permute(1, 2, 0))
    axs[1, 0].set_title("Random Scale - Img 1")
    axs[1, 1].imshow(res_random[1].permute(1, 2, 0))
    axs[1, 1].set_title("Random Scale - Img 2\n(Different Size & Pos!)")
    
    for ax in axs.flatten():
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()
    print("RandomErase test done.")