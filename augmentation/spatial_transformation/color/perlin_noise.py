# -*- coding: utf-8 -*-
"""
Perlin/Fractal Noise Augmentation.

Generates multi-octave fractal noise natively in PyTorch for high-performance
data augmentation pipelines.

@author: Diyar Altinses, M.Sc.
"""

from typing import Tuple, Union
import torch

class PerlinNoise(torch.nn.Module):
    """
    Applies Fractal/Perlin Noise to the input tensor.
    Supports both grayscale and color noise, and operates fully on GPU.
    """

    def __init__(
        self, 
        probability: float = 0.5, 
        octaves: Union[int, Tuple[int, int]] = (2, 5),
        persistence: float = 0.5, 
        frequency: Union[int, Tuple[int, int]] = (2, 6),
        alpha: Union[float, Tuple[float, float]] = (0.3, 0.6),
        color: bool = False
    ):
        """
        Parameters
        ----------
        probability : float
            Probability of applying the transform (0.0 to 1.0).
        octaves : int or tuple
            Number of layers of noise to combine. More octaves = more detail.
            If tuple, a random integer is chosen from the range.
        persistence : float
            Scaling factor for amplitude of higher octaves (usually 0.5).
        frequency : int or tuple
            Base frequency (roughness) of the noise.
            If tuple, a random integer is chosen from the range.
        alpha : float or tuple
            Blending factor (transparency) of the noise. 0.0 = Invisible, 1.0 = Full Noise.
            If tuple, sampled uniformly from range.
        color : bool
            If True, generates independent noise for each RGB channel (colorful).
            If False, generates same noise for all channels (grayscale/structure).
        """
        super().__init__()
        self.probability = probability
        self.octaves = octaves
        self.persistence = persistence
        self.frequency = frequency
        self.alpha = alpha
        self.color = color

    def extra_repr(self) -> str:
        return f"probability={self.probability}, octaves={self.octaves}, color={self.color}, alpha={self.alpha}"

    def _get_value(self, param: Union[int, float, Tuple, list], is_int: bool = False):
        """Helper to sample random values from tuples or return scalars."""
        if isinstance(param, (tuple, list)):
            if is_int:
                return torch.randint(low=param[0], high=param[1] + 1, size=(1,)).item()
            else:
                return torch.empty(1).uniform_(param[0], param[1]).item()
        return param

    def generate_fractal_noise(
        self, 
        shape: torch.Size, 
        octaves: int, 
        freq: int, 
        device: torch.device
    ) -> torch.Tensor:
        """
        Generates fractal noise using bicubic upsampling (Fast Perlin Approximation).
        """
        # Batch size, Channels, Height, Width
        B, C, H, W = shape
        
        # If not colorful, we only need 1 channel of noise which we broadcast later
        noise_channels = C if self.color else 1
        
        noise = torch.zeros((B, noise_channels, H, W), device=device)
        amplitude = 1.0
        current_freq = freq

        for _ in range(octaves):
            # 1. Generate random noise at lower resolution
            # We add +2 to buffer against boundary artifacts during interpolation
            noise_low = torch.rand(
                (B, noise_channels, current_freq + 2, current_freq + 2), 
                device=device
            )
            
            # 2. Upsample to target resolution using Bicubic interpolation (smooths the random pixels)
            # align_corners=False usually gives better "texture" looking results
            noise_upsampled = torch.nn.functional.interpolate(
                noise_low, size=(H, W), mode='bicubic', align_corners=False
            )
            
            # 3. Accumulate
            noise += amplitude * noise_upsampled
            
            # 4. Step
            amplitude *= self.persistence
            current_freq *= 2 # Lacunarity is usually fixed at 2 for Perlin
            
        return noise

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): Input image (N, C, H, W) or (C, H, W).
            
        Returns:
            torch.Tensor: Noised image.
        """
        # 1. Probability Check
        if torch.rand(1, device=img.device) > self.probability:
            return img

        # 2. Handle Inputs (Standardize to Batch N, C, H, W)
        is_batch = img.ndim == 4
        x = img if is_batch else img.unsqueeze(0)
        
        B, C, H, W = x.shape
        device = x.device

        # 3. Sample Dynamic Parameters
        current_octaves = self._get_value(self.octaves, is_int=True)
        current_freq = self._get_value(self.frequency, is_int=True)
        current_alpha = self._get_value(self.alpha, is_int=False)

        # 4. Generate Noise
        # Noise shape will be (B, C, H, W) or (B, 1, H, W)
        noise = self.generate_fractal_noise(
            shape=(B, C, H, W), 
            octaves=current_octaves, 
            freq=current_freq, 
            device=device
        )
        
        # 5. Normalize Noise to [0, 1] range to match image
        # (Min-Max Normalization per batch item is safer)
        flat = noise.view(B, -1)
        vmin = flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        vmax = flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        noise = (noise - vmin) / (vmax - vmin + 1e-6)

        # 6. Normalize Input if necessary (Assume float input is 0-1, byte is 0-255)
        # We process in 0-1 float space
        was_byte = False
        if x.dtype == torch.uint8 or x.max() > 1.0:
            x = x.float() / 255.0
            was_byte = True

        # 7. Blend
        # Broadcasting handles (B, 1, H, W) noise against (B, 3, H, W) image automatically
        out = (1 - current_alpha) * x + (current_alpha * noise)

        # 8. Restore Type
        if was_byte:
            out = (out * 255).clamp(0, 255).to(torch.uint8)
        else:
            out = out.clamp(0, 1)

        return out if is_batch else out.squeeze(0)


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # Create dummy data: 2 images, 3 channels (RGB), 256x256
    image_batch = torch.zeros(2, 3, 256, 256)
    # Make image 1 slightly grey, image 2 slightly brighter for contrast
    image_batch[0] = 0.2
    image_batch[1] = 0.5
    
    # Test 1: Grayscale Noise (Structure)
    aug_gray = PerlinNoise(probability=1.0, color=False, alpha=0.5, frequency=(4, 8))
    out_gray = aug_gray(image_batch.clone())

    # Test 2: Color Noise (RGB Distortion)
    aug_color = PerlinNoise(probability=1.0, color=True, alpha=0.6, frequency=3)
    out_color = aug_color(image_batch.clone())

    # Visualization
    fig, axs = plt.subplots(2, 3, figsize=(10, 7))
    
    # Row 1: Image 1
    axs[0, 0].imshow(image_batch[0].permute(1, 2, 0))
    axs[0, 0].set_title("Original (Val=0.2)")
    
    axs[0, 1].imshow(out_gray[0].permute(1, 2, 0))
    axs[0, 1].set_title("Grayscale Noise")
    
    axs[0, 2].imshow(out_color[0].permute(1, 2, 0))
    axs[0, 2].set_title("Color Noise")

    # Row 2: Image 2
    axs[1, 0].imshow(image_batch[1].permute(1, 2, 0))
    axs[1, 0].set_title("Original (Val=0.5)")
    
    axs[1, 1].imshow(out_gray[1].permute(1, 2, 0))
    
    axs[1, 2].imshow(out_color[1].permute(1, 2, 0))

    for ax in axs.flatten():
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()
    print("Test finished successfully.")