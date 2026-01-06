# -*- coding: utf-8 -*-
"""
Additive Noise Augmentations.

Simulates sensor thermal noise, ISO grain, or transmission interference.
Equation: Output = Input + Noise (Independent of signal strength)

@author: Diyar Altinses, M.Sc.
"""

from typing import Union, Tuple
import torch


class AdditiveGaussianNoise(torch.nn.Module):
    """
    Applies Additive Gaussian Noise.
    Also known as "Normal Noise". Matches the distribution of electronic circuit noise.
    
    Formula: Out = In + Gaussian(mean, std)
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
            Mean of the distribution (usually 0.0).
        std : float or tuple
            Standard deviation (strength). 
            If tuple, a random strength is chosen per forward pass.
        """
        super().__init__()
        self.probability = probability
        self.mean = mean
        self.std = std

    def extra_repr(self) -> str:
        return f"probability={self.probability}, mean={self.mean}, std={self.std}"

    def forward(self, img: torch.Tensor) -> torch.Tensor:
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

        # 4. Generate Additive Noise
        # randn_like generates values from Normal distribution N(0, 1)
        noise = torch.randn_like(x) * sigma + self.mean
        
        # 5. Apply Additive Formula: x + noise
        out = x + noise

        # 6. Clamp and Restore
        if not is_float:
            out = out.clamp(0, 255).to(img.dtype)
        else:
            out = out.clamp(0, 1)

        return out


class AdditiveWhiteNoise(torch.nn.Module):
    """
    Applies Additive White Noise (Uniform Distribution).
    
    Formula: Out = In + Uniform(-amp, amp)
    Unlike Gaussian, this noise is bounded strictly within a range.
    """

    def __init__(self, probability: float = 1.0, amplitude: float = 0.1):
        """
        Parameters
        ----------
        probability : float
            Probability of applying the noise.
        amplitude : float
            The noise will be sampled uniformly from [-amplitude, +amplitude].
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

        # Generate Uniform Noise
        # torch.rand generates [0, 1]
        # (rand - 0.5) * 2 -> [-1, 1]
        # * amplitude -> [-amp, amp]
        noise = (torch.rand_like(x) - 0.5) * 2 * self.amplitude
        
        # Apply Additive Formula
        out = x + noise

        # Clamp and Restore
        if not is_float:
            out = out.clamp(0, 255).to(img.dtype)
        else:
            out = out.clamp(0, 1)

        return out


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # Create a gradient image (0 to 1)
    H, W = 100, 256
    grad = torch.linspace(0, 1, W).view(1, W).expand(H, W).unsqueeze(0)
    
    # 1. Additive Gaussian
    aug_gauss = AdditiveGaussianNoise(probability=1.0, mean=0.0, std=0.2)
    out_gauss = aug_gauss(grad.clone())

    # 2. Additive White (Uniform)
    aug_white = AdditiveWhiteNoise(probability=1.0, amplitude=0.4)
    out_white = aug_white(grad.clone())

    # Visualization
    
    

    fig, axs = plt.subplots(3, 1, figsize=(8, 8))
    
    axs[0].imshow(grad[0], cmap='gray', vmin=0, vmax=1)
    axs[0].set_title("Original (Gradient)")
    axs[0].axis('off')
    
    axs[1].imshow(out_gauss[0], cmap='gray', vmin=0, vmax=1)
    axs[1].set_title("Additive Gaussian (Visible everywhere, even in black areas)")
    axs[1].axis('off')

    axs[2].imshow(out_white[0], cmap='gray', vmin=0, vmax=1)
    axs[2].set_title("Additive White/Uniform (Grainy everywhere)")
    axs[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    print("Additive Noise test done.")