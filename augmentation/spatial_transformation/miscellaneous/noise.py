# -*- coding: utf-8 -*-
"""
Gaussian Noise Augmentation.

Adds random Gaussian (Normal) noise to the tensor.
Useful for simulating sensor noise (ISO grain) or improving robustness.

@author: Diyar Altinses, M.Sc.
"""

from typing import Optional, Tuple, Union
import torch


class GaussianNoise(torch.nn.Module):
    """
    Adds Gaussian noise to the input tensor.
    Formula: output = input + N(mean, std)
    """

    def __init__(
        self, 
        probability: float = 0.5, 
        mean: float = 0.0, 
        std: Union[float, Tuple[float, float]] = 0.1, 
        clamp: Optional[Tuple[float, float]] = None
    ):
        """
        Parameters
        ----------
        probability : float
            Probability of applying the noise.
        mean : float
            Mean of the Gaussian distribution (usually 0).
        std : float or tuple
            Standard deviation (strength) of the noise.
            If float: fixed value.
            If tuple (min, max): random value selected uniformly from this range per call.
        clamp : tuple, optional
            (min, max) to clamp the output values. 
            Example: (0.0, 1.0) for standard images to prevent overflow.
        """
        super().__init__()
        self.probability = probability
        self.mean = mean
        self.std = std
        self.clamp_range = clamp

    def extra_repr(self) -> str:
        return f"probability={self.probability}, mean={self.mean}, std={self.std}, clamp={self.clamp_range}"

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): Input image/signal.
            
        Returns:
            torch.Tensor: Noisy image.
        """
        # 1. Probability Check
        if torch.rand(1, device=img.device) > self.probability:
            return img

        # 2. Determine Noise Strength (std)
        # Allows for dynamic noise levels during training
        if isinstance(self.std, (tuple, list)):
            # Randomly pick a std between min and max
            sigma = torch.empty(1, device=img.device).uniform_(self.std[0], self.std[1])
        else:
            sigma = self.std

        # 3. Generate Noise
        # noise = torch.randn_like(img) creates noise on the same device and dtype as img
        # However, if img is Byte/Int, we need float noise.
        
        if img.is_floating_point():
            noise = torch.randn_like(img) * sigma + self.mean
            out = img + noise
        else:
            # If input is int/byte (e.g. 0-255), we must convert to float for math
            x_float = img.float()
            noise = torch.randn_like(x_float) * sigma + self.mean
            out = x_float + noise
            # We don't cast back immediately to allow clamping first

        # 4. Clamp (Optional but recommended for images)
        if self.clamp_range is not None:
            out = torch.clamp(out, min=self.clamp_range[0], max=self.clamp_range[1])

        # 5. Restore Type if input was not float
        if not img.is_floating_point():
            out = out.to(img.dtype)

        return out


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # 1. Create a clean gradient image
    H, W = 100, 256
    img = torch.linspace(0, 1, W).view(1, W).expand(H, W).unsqueeze(0) # (1, H, W)
    
    # 2. Augment with variable noise strength
    # std between 0.05 and 0.2
    aug = GaussianNoise(probability=1.0, mean=0.0, std=(0.05, 0.2), clamp=(0.0, 1.0))
    
    out = aug(img)

    # Visualization
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    
    axs[0].imshow(img[0], cmap='gray', vmin=0, vmax=1)
    axs[0].set_title("Original (Clean)")
    axs[0].axis('off')
    
    axs[1].imshow(out[0], cmap='gray', vmin=0, vmax=1)
    axs[1].set_title("Augmented (Gaussian Noise)")
    axs[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    print("GaussianNoise test done.")