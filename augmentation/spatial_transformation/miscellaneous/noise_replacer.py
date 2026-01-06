# -*- coding: utf-8 -*-
"""
Replace Noise Augmentations.

Completely discards the input signal and replaces it with pure noise.
Useful for simulating signal loss, sensor failure, or testing network robustness to pure noise.

@author: Diyar Altinses, M.Sc.
"""

from typing import Optional, Tuple
import torch


class ReplaceGaussianNoise(torch.nn.Module):
    """
    Replaces the input entirely with Gaussian Noise.
    Formula: Out = N(0, 1) * max_amplitude
    """

    def __init__(
        self, 
        probability: float = 1.0, 
        scale_by_input_max: bool = True,
        clamp: Optional[Tuple[float, float]] = None
    ):
        """
        Parameters
        ----------
        probability : float
            Probability of replacing the input.
        scale_by_input_max : bool
            If True, the noise amplitude matches the max value of the original input.
            If False, produces standard N(0,1) noise.
        clamp : tuple, optional
            (min, max) to clamp the output values.
        """
        super().__init__()
        self.probability = probability
        self.scale_by_input_max = scale_by_input_max
        self.clamp_range = clamp

    def extra_repr(self) -> str:
        return f"probability={self.probability}, scale_input={self.scale_by_input_max}, clamp={self.clamp_range}"

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # 1. Probability Check
        if torch.rand(1, device=img.device) > self.probability:
            return img

        # 2. Generate Noise
        # randn_like ensures correct device and shape
        noise = torch.randn_like(img, dtype=torch.float32)

        # 3. Scale Noise
        if self.scale_by_input_max:
            # Scale noise to match the range of the input image
            # Avoid multiplying by 0 if image is empty
            max_val = img.max()
            scale = max_val if max_val > 0 else 1.0
            noise = noise * scale

        # 4. Clamp (Optional)
        if self.clamp_range is not None:
            noise = torch.clamp(noise, min=self.clamp_range[0], max=self.clamp_range[1])

        # 5. Restore Type
        if not img.is_floating_point():
            noise = noise.to(img.dtype)

        return noise


class ReplaceWhiteNoise(torch.nn.Module):
    """
    Replaces the input entirely with Uniform (White) Noise.
    Formula: Out = Uniform(0, 1) * max_amplitude
    """

    def __init__(
        self, 
        probability: float = 1.0, 
        scale_by_input_max: bool = True,
        clamp: Optional[Tuple[float, float]] = None
    ):
        """
        Parameters
        ----------
        probability : float
            Probability of replacing the input.
        scale_by_input_max : bool
            If True, noise scales to input max.
        clamp : tuple, optional
            (min, max) range.
        """
        super().__init__()
        self.probability = probability
        self.scale_by_input_max = scale_by_input_max
        self.clamp_range = clamp

    def extra_repr(self) -> str:
        return f"probability={self.probability}, scale_input={self.scale_by_input_max}, clamp={self.clamp_range}"

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(1, device=img.device) > self.probability:
            return img

        # 1. Generate Uniform Noise [0, 1]
        noise = torch.rand_like(img, dtype=torch.float32)

        # 2. Scale Noise
        if self.scale_by_input_max:
            max_val = img.max()
            scale = max_val if max_val > 0 else 1.0
            noise = noise * scale

        # 3. Clamp
        if self.clamp_range is not None:
            noise = torch.clamp(noise, min=self.clamp_range[0], max=self.clamp_range[1])

        # 4. Restore Type
        if not img.is_floating_point():
            noise = noise.to(img.dtype)

        return noise


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # Create a dummy image
    img = torch.ones(1, 100, 100) * 0.5
    
    # 1. Replace with Gaussian
    aug_gauss = ReplaceGaussianNoise(probability=1.0, clamp=(0, 1))
    out_gauss = aug_gauss(img)

    # 2. Replace with White Noise
    aug_white = ReplaceWhiteNoise(probability=1.0, clamp=(0, 1))
    out_white = aug_white(img)

    # Visualization
    
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    
    axs[0].imshow(img[0], cmap='gray', vmin=0, vmax=1)
    axs[0].set_title("Original (Solid Gray)")
    axs[0].axis('off')
    
    axs[1].imshow(out_gauss[0], cmap='gray', vmin=0, vmax=1)
    axs[1].set_title("Replaced with Gaussian")
    axs[1].axis('off')

    axs[2].imshow(out_white[0], cmap='gray', vmin=0, vmax=1)
    axs[2].set_title("Replaced with White Noise")
    axs[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    print("Replace Noise test done.")