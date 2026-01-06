# -*- coding: utf-8 -*-
"""
Multiplicative Noise Augmentations (Speckle Noise).

Simulates signal-dependent noise often found in Radar, SAR, or Ultrasound imagery.
Equation: Output = Input + (Input * Noise)

@author: Diyar Altinses, M.Sc.
"""

from typing import Union, Tuple
import torch


class MultiplicativeGaussianNoise(torch.nn.Module):
    """
    Applies Multiplicative Gaussian Noise (Speckle Noise).
    
    Formula: Out = In + (In * Gaussian(mean, std))
    This models noise that increases with signal strength (brighter areas = more noise).
    """

    def __init__(
        self, 
        probability: float = 1.0, 
        mean: float = 0.0, 
        std: Union[float, Tuple[float, float]] = 0.1
    ):
        """
        Parameters
        ----------
        probability : float
            Probability of applying the noise.
        mean : float
            Mean of the Gaussian distribution (usually 0).
        std : float or tuple
            Standard deviation (strength) of the noise factor.
            If tuple, a random value is chosen per call.
        """
        super().__init__()
        self.probability = probability
        self.mean = mean
        self.std = std

    def extra_repr(self) -> str:
        return f"probability={self.probability}, mean={self.mean}, std={self.std}"

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): Input image.
            
        Returns:
            torch.Tensor: Image with speckle noise.
        """
        # 1. Probability Check
        if torch.rand(1, device=img.device) > self.probability:
            return img

        # 2. Determine Noise Strength
        if isinstance(self.std, (tuple, list)):
            sigma = torch.empty(1, device=img.device).uniform_(self.std[0], self.std[1])
        else:
            sigma = self.std

        # 3. Handle Float vs Byte
        is_float = img.is_floating_point()
        x = img if is_float else img.float()

        # 4. Generate Noise: N(mean, std)
        # We use randn_like to match device/shape automatically
        noise = torch.randn_like(x) * sigma + self.mean
        
        # 5. Apply Multiplicative Formula: x + x * noise
        out = x + (x * noise)

        # 6. Clamp and Restore
        if not is_float:
            out = out.clamp(0, 255).to(img.dtype)
        else:
            out = out.clamp(0, 1) # Assuming standard float images 0-1

        return out


class MultiplicativeWhiteNoise(torch.nn.Module):
    """
    Applies Multiplicative Uniform Noise.
    
    Formula: Out = In + (In * Uniform(-amp, amp))
    The noise is sampled from a uniform distribution rather than Gaussian.
    """

    def __init__(self, probability: float = 1.0, amplitude: float = 0.1):
        """
        Parameters
        ----------
        probability : float
            Probability of applying the noise.
        amplitude : float
            Range of the uniform noise factor. 
            Noise will be in range [-amplitude, amplitude].
        """
        super().__init__()
        self.probability = probability
        self.amplitude = amplitude

    def extra_repr(self) -> str:
        return f"probability={self.probability}, amplitude={self.amplitude}"

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(1, device=img.device) > self.probability:
            return img

        is_float = img.is_floating_point()
        x = img if is_float else img.float()

        # Generate Uniform Noise between [-amplitude, amplitude]
        # (rand - 0.5) * 2 * amp  -> range [-amp, amp]
        noise = (torch.rand_like(x) - 0.5) * 2 * self.amplitude
        
        # Apply Multiplicative Formula
        out = x + (x * noise)

        if not is_float:
            out = out.clamp(0, 255).to(img.dtype)
        else:
            out = out.clamp(0, 1)

        return out


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # Create a gradient image (0 to 1)
    # This helps visualize that multiplicative noise is stronger in bright areas (right side)
    # and weak/zero in dark areas (left side).
    H, W = 100, 256
    grad = torch.linspace(0, 1, W).view(1, W).expand(H, W).unsqueeze(0)
    
    # 1. Multiplicative Gaussian (Speckle)
    aug_gauss = MultiplicativeGaussianNoise(probability=1.0, mean=0.0, std=0.2)
    out_gauss = aug_gauss(grad.clone())

    # 2. Multiplicative White
    aug_white = MultiplicativeWhiteNoise(probability=1.0, amplitude=0.3)
    out_white = aug_white(grad.clone())

    # Visualization
    fig, axs = plt.subplots(3, 1, figsize=(8, 8))
    
    axs[0].imshow(grad[0], cmap='gray', vmin=0, vmax=1)
    axs[0].set_title("Original (Gradient)")
    axs[0].axis('off')
    
    axs[1].imshow(out_gauss[0], cmap='gray', vmin=0, vmax=1)
    axs[1].set_title("Multiplicative Gaussian (Note: Noise is stronger on the right!)")
    axs[1].axis('off')

    axs[2].imshow(out_white[0], cmap='gray', vmin=0, vmax=1)
    axs[2].set_title("Multiplicative White Noise")
    axs[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    print("Multiplicative Noise test done.")