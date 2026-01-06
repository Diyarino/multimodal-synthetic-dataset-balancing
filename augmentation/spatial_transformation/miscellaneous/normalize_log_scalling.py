# -*- coding: utf-8 -*-
"""
Gamma Correction Augmentation.

Adjusts the luminance of an image non-linearly.
Standard technique for changing brightness/contrast in photography.

@author: Diyar Altinses, M.Sc.
"""

from typing import Union, Tuple
import torch


class GammaCorrection(torch.nn.Module):
    """
    Applies Gamma Correction to the input image.
    Formula: Out = In ^ gamma
    
    Gamma < 1.0: Image gets brighter (expands dark areas).
    Gamma > 1.0: Image gets darker (compresses dark areas).
    """

    def __init__(self, probability: float = 0.5, gamma: Union[float, Tuple[float, float]] = (0.5, 2.0)):
        """
        Parameters
        ----------
        probability : float
            Probability of applying the transform.
        gamma : float or tuple
            The exponent value. 
            If tuple, a random value is chosen from the range.
            Typical values: 0.5 to 2.0.
        """
        super().__init__()
        self.probability = probability
        self.gamma_range = gamma

    def extra_repr(self) -> str:
        return f"probability={self.probability}, gamma_range={self.gamma_range}"

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # 1. Probability Check
        if torch.rand(1, device=img.device) > self.probability:
            return img

        # 2. Determine Gamma Value
        if isinstance(self.gamma_range, (tuple, list)):
            # Random uniform selection
            gamma = torch.empty(1, device=img.device).uniform_(self.gamma_range[0], self.gamma_range[1])
        else:
            gamma = self.gamma_range

        # 3. Apply Power Law
        # Values must be in range [0, 1] for gamma to work correctly visually.
        # If input is uint8, we usually normalize -> power -> denormalize, 
        # or just apply directly if we are careful. Best practice is float [0,1].
        
        is_float = img.is_floating_point()
        
        # If byte (0-255), normalize to 0-1 first
        if not is_float:
            x = img.float() / 255.0
        elif img.max() > 1.0:
             x = img / 255.0 # Assume it's float but 0-255 scale
        else:
            x = img

        # Safe power (add epsilon to avoid derivative issues at 0)
        out = torch.pow(x + 1e-6, gamma)

        # 4. Restore Original Range/Type
        if not is_float:
            out = (out * 255.0).clamp(0, 255).to(img.dtype)
        elif img.max() > 1.0:
            out = out * 255.0
        else:
            out = out.clamp(0, 1)

        return out


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # Create a linear gradient (0 to 1)
    # This shows clearly how gamma bends the curve
    W = 256
    grad = torch.linspace(0, 1, W).view(1, W).expand(100, W).unsqueeze(0)
    
    # 1. Brighter (Gamma < 1)
    aug_bright = GammaCorrection(probability=1.0, gamma=0.5)
    out_bright = aug_bright(grad)

    # 2. Darker (Gamma > 1)
    aug_dark = GammaCorrection(probability=1.0, gamma=2.0)
    out_dark = aug_dark(grad)

    # Visualization
    

    fig, axs = plt.subplots(3, 1, figsize=(8, 8))
    
    axs[0].imshow(grad[0], cmap='gray', vmin=0, vmax=1)
    axs[0].set_title("Original (Linear Gradient)")
    axs[0].axis('off')
    
    axs[1].imshow(out_bright[0], cmap='gray', vmin=0, vmax=1)
    axs[1].set_title("Gamma 0.5 (Brighter / Bleached)")
    axs[1].axis('off')

    axs[2].imshow(out_dark[0], cmap='gray', vmin=0, vmax=1)
    axs[2].set_title("Gamma 2.0 (Darker / Contrast)")
    axs[2].axis('off')
    
    plt.tight_layout()
    plt.show()