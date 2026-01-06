# -*- coding: utf-8 -*-
"""
Salt and Pepper Noise Augmentation.

Replaces random pixels with the minimum (Pepper) or maximum (Salt) value.
Simulates "dead pixels" (always black) or "hot pixels" (always white).

@author: Diyar Altinses, M.Sc.
"""

import torch


class SaltAndPepperNoise(torch.nn.Module):
    """
    Applies Salt and Pepper noise to the input image.
    
    - Salt: Randomly sets pixels to the maximum value (White).
    - Pepper: Randomly sets pixels to the minimum value (Black).
    """

    def __init__(
        self, 
        probability: float = 1.0, 
        density: float = 0.05, 
        salt_ratio: float = 0.5
    ):
        """
        Parameters
        ----------
        probability : float
            Probability of applying the augmentation.
        density : float
            Total percentage of pixels to replace (0.0 to 1.0).
            Example: 0.05 means 5% of the image will be noise.
        salt_ratio : float
            Ratio of Salt (White) vs Pepper (Black).
            0.5 = Equal amount of white and black dots.
            1.0 = Only Salt (White dots).
            0.0 = Only Pepper (Black dots).
        """
        super().__init__()
        self.probability = probability
        self.density = density
        self.salt_ratio = salt_ratio

    def extra_repr(self) -> str:
        return f"probability={self.probability}, density={self.density}, salt_ratio={self.salt_ratio}"

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): Input image.
            
        Returns:
            torch.Tensor: Noisy image.
        """
        # 1. Probability Check
        if torch.rand(1, device=img.device) > self.probability:
            return img

        # 2. Setup
        # Clone to avoid modifying the input in-place
        out = img.clone()
        
        # Determine dynamic range (Min/Max)
        # For batches, this assumes all images share the range (e.g. 0-1 or 0-255).
        # We can detect per image, but global max/min is safer for standard S&P behavior.
        min_val = img.min()
        max_val = img.max()

        # 3. Generate Noise Mask
        # We create one random matrix for efficiency
        # rand values are in [0, 1)
        noise_map = torch.rand_like(out)
        
        # 4. Apply Salt (White)
        # If density is 0.1 and ratio is 0.5 -> We want top 5% pixels to be Salt.
        # Threshold: < (density * salt_ratio)
        salt_threshold = self.density * self.salt_ratio
        out[noise_map < salt_threshold] = max_val
        
        # 5. Apply Pepper (Black)
        # We want the *other* 5% to be Pepper.
        # We take pixels from the upper end of the random distribution to avoid overlapping with Salt.
        # Threshold: > 1 - (density * (1 - salt_ratio))
        pepper_count = self.density * (1.0 - self.salt_ratio)
        pepper_threshold = 1.0 - pepper_count
        out[noise_map > pepper_threshold] = min_val
        
        return out


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # Create a smooth gray gradient image
    H, W = 100, 100
    img = torch.linspace(0.2, 0.8, W).view(1, W).expand(H, W).unsqueeze(0).unsqueeze(0)
    
    # 1. Balanced Salt & Pepper
    aug_balanced = SaltAndPepperNoise(probability=1.0, density=0.1, salt_ratio=0.5)
    out_balanced = aug_balanced(img)

    # 2. Only Pepper (Dead Pixels)
    aug_pepper = SaltAndPepperNoise(probability=1.0, density=0.1, salt_ratio=0.0)
    out_pepper = aug_pepper(img)

    # Visualization
    

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    
    axs[0].imshow(img[0, 0], cmap='gray', vmin=0, vmax=1)
    axs[0].set_title("Original")
    axs[0].axis('off')
    
    axs[1].imshow(out_balanced[0, 0], cmap='gray', vmin=0, vmax=1)
    axs[1].set_title("Salt & Pepper\n(Density=0.1, Ratio=0.5)")
    axs[1].axis('off')

    axs[2].imshow(out_pepper[0, 0], cmap='gray', vmin=0, vmax=1)
    axs[2].set_title("Only Pepper\n(Black Dots)")
    axs[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    print("SaltAndPepperNoise test done.")