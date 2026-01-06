# -*- coding: utf-8 -*-
"""
Spectral Gain / Magnitude Manipulation.

Randomly scales the magnitude of specific frequency components.
This acts as a "Spectral Jitter" or random EQ, changing the timbre (1D) or texture (2D).

@author: Diyar Altinses, M.Sc.
"""

from typing import Tuple
import torch


class SpectralGain(torch.nn.Module):
    """
    Randomly modifies the magnitude of frequency components.
    
    The phase is preserved, so the structure (edges/timing) remains roughly the same,
    but the intensity of certain frequencies changes.
    """

    def __init__(
        self, 
        probability: float = 1.0, 
        gain_range: Tuple[float, float] = (0.5, 1.5), 
        fraction: float = 0.2
    ):
        """
        Parameters
        ----------
        probability : float
            Probability of applying the augmentation.
        gain_range : tuple
            Min and Max scaling factor for the magnitude.
            Example: (0.5, 1.5) means magnitude can be halved or increased by 50%.
        fraction : float
            Percentage of frequency bins to modify (0.0 to 1.0).
            Replaces the fixed 'features' count for better scalability.
        """
        super().__init__()
        self.probability = probability
        self.gain_range = gain_range
        self.fraction = fraction

    def extra_repr(self) -> str:
        return f"probability={self.probability}, gain_range={self.gain_range}, fraction={self.fraction}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor (N, C, L) or (N, C, H, W).
            
        Returns:
            torch.Tensor: Augmented tensor in time/spatial domain.
        """
        # 1. Probability Check
        if torch.rand(1, device=x.device) > self.probability:
            return x

        # 2. Determine Dimensions
        if x.ndim == 3:
            fft_dims = (-1,)
        elif x.ndim == 4:
            fft_dims = (-2, -1)
        else:
            return x

        # 3. FFT (Real input -> Complex spectrum)
        # Use rfftn for efficiency with real-valued inputs
        x_freq = torch.fft.rfftn(x, dim=fft_dims)
        
        # Decompose into Magnitude and Phase
        mag = torch.abs(x_freq)
        phase = torch.angle(x_freq)

        # 4. Generate Gain Mask (Vectorized)
        # Create a mask of which frequencies to modify
        # shape matches magnitude spectrum
        rand_mask = torch.rand_like(mag) < self.fraction
        
        # Generate random gain values for the selected frequencies
        # Gains are uniformly distributed in [min, max]
        gains = torch.empty_like(mag).uniform_(self.gain_range[0], self.gain_range[1])
        
        # Apply gain only where mask is True
        # Original magnitude is kept where mask is False (gain=1.0)
        final_gains = torch.where(rand_mask, gains, torch.ones_like(mag))
        
        # 5. Modify Magnitude
        mag_modified = mag * final_gains

        # 6. Reconstruct Complex Tensor
        x_modified = torch.polar(mag_modified, phase)

        # 7. Inverse FFT
        # irfftn automatically returns real output from rfftn input
        out = torch.fft.irfftn(x_modified, s=x.shape[fft_dims[0]:], dim=fft_dims)

        return out


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # ---------------------------------------------------------
    # Test 1: 1D Signal (Mix of Frequencies)
    # ---------------------------------------------------------
    t = torch.linspace(0, 1, 200)
    # Base signal: Low freq + High freq noise
    signal = (torch.sin(20 * t) + 0.3 * torch.sin(100 * t)).view(1, 1, -1)
    
    # Augment: Modify 50% of frequencies with gains between 0.0 and 2.0
    aug = SpectralGain(probability=1.0, gain_range=(0.0, 3.0), fraction=0.5)
    out_1d = aug(signal.clone())

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------
    

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    
    # Time Domain
    ax[0].plot(signal[0, 0].numpy(), label="Original", alpha=0.8)
    ax[0].plot(out_1d[0, 0].numpy(), label="Augmented", alpha=0.8)
    ax[0].set_title("Time Domain")
    ax[0].legend()
    
    # Frequency Domain Check
    fft_orig = torch.fft.rfft(signal).abs()[0, 0]
    fft_aug = torch.fft.rfft(out_1d).abs()[0, 0]
    
    ax[1].plot(fft_orig.numpy(), label="Original Spectrum")
    ax[1].plot(fft_aug.numpy(), label="Augmented Spectrum", alpha=0.6)
    ax[1].set_title("Frequency Domain (Magnitude)\nNote random peaks scaling")
    ax[1].legend()
    
    plt.tight_layout()
    plt.show()
    print("SpectralGain test done.")