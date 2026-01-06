# -*- coding: utf-8 -*-
"""
Spectral Normalization Augmentation.

Rescales the Magnitude Spectrum of a signal/image to a specific range.
Useful for standardizing the energy/loudness of inputs in the frequency domain.

@author: Diyar Altinses, M.Sc.
"""

from typing import Tuple
import torch


class SpectralNormalization(torch.nn.Module):
    """
    Normalizes the Magnitude Spectrum to a target range [min, max].
    
    This preserves the relative structure of frequencies (peaks remain peaks)
    but stretches/compresses the overall energy distribution.
    """

    def __init__(
        self, 
        probability: float = 1.0, 
        target_range: Tuple[float, float] = (0.0, 1.0),
        per_channel: bool = True
    ):
        """
        Parameters
        ----------
        probability : float
            Probability of applying the augmentation.
        target_range : tuple
            (min, max) target values for the magnitude spectrum.
            Default (0.0, 1.0).
        per_channel : bool
            If True, calculates min/max for each channel independently.
            If False, normalizes based on the global min/max of the sample.
        """
        super().__init__()
        self.probability = probability
        self.min_val, self.max_val = target_range
        self.per_channel = per_channel

    def extra_repr(self) -> str:
        return f"probability={self.probability}, target_range=({self.min_val}, {self.max_val})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor (N, C, L) or (N, C, H, W).
            
        Returns:
            torch.Tensor: Normalized tensor in time/spatial domain.
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
        x_freq = torch.fft.rfftn(x, dim=fft_dims)
        
        mag = torch.abs(x_freq)
        phase = torch.angle(x_freq)

        # 4. Calculate Min/Max (Vectorized)
        # We need to find min/max along the frequency dimensions (last dims)
        # Flatten frequency dims to make min/max calculation easy
        # Shape: (N, C, Freq_Bins)
        flatten_dim = 2 # Keep N(0) and C(1) intact
        mag_flat = mag.flatten(start_dim=flatten_dim)
        
        if self.per_channel:
            # Min/Max per channel: (N, C, 1)
            current_min = mag_flat.min(dim=-1, keepdim=True)[0]
            current_max = mag_flat.max(dim=-1, keepdim=True)[0]
        else:
            # Global Min/Max per sample: (N, 1, 1)
            # Flatten channels too for calculation
            mag_sample = mag.flatten(start_dim=1)
            current_min = mag_sample.min(dim=-1, keepdim=True)[0].unsqueeze(1)
            current_max = mag_sample.max(dim=-1, keepdim=True)[0].unsqueeze(1)

        # Reshape stats to match original 'mag' shape for broadcasting
        # We append 1s for the frequency dimensions
        shape_diff = mag.ndim - current_min.ndim
        for _ in range(shape_diff):
            current_min = current_min.unsqueeze(-1)
            current_max = current_max.unsqueeze(-1)

        # 5. Apply Min-Max Normalization
        # Formula: (X - min) / (max - min + eps) * (tgt_max - tgt_min) + tgt_min
        numerator = mag - current_min
        denominator = (current_max - current_min) + 1e-8 # Epsilon against div/0
        
        mag_norm = numerator / denominator
        
        scale = self.max_val - self.min_val
        mag_scaled = mag_norm * scale + self.min_val

        # 6. Reconstruct
        x_modified = torch.polar(mag_scaled, phase)
        out = torch.fft.irfftn(x_modified, s=x.shape[fft_dims[0]:], dim=fft_dims)

        return out


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # ---------------------------------------------------------
    # Test: 1D Signal with varying amplitude peaks
    # ---------------------------------------------------------
    t = torch.linspace(0, 1, 200)
    # Signal with a HUGE peak at 10Hz and a TINY peak at 50Hz
    signal = (100.0 * torch.sin(20 * t) + 1.0 * torch.sin(100 * t)).view(1, 1, -1)
    
    # Augment: Force spectrum to be between 0.0 and 1.0
    # This compresses the huge peak and relatively boosts the floor
    aug = SpectralNormalization(probability=1.0, target_range=(0.0, 1.0))
    out = aug(signal.clone())

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------
    # Compute Spectrums for display
    spec_orig = torch.fft.rfft(signal).abs().flatten()
    spec_norm = torch.fft.rfft(out).abs().flatten()

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    
    # Time Domain
    axs[0, 0].plot(signal[0, 0].numpy())
    axs[0, 0].set_title("Original Time (High Amplitude)")
    
    axs[0, 1].plot(out[0, 0].numpy(), color='orange')
    axs[0, 1].set_title("Normalized Time (Rescaled)")
    
    # Frequency Domain
    axs[1, 0].plot(spec_orig.numpy())
    axs[1, 0].set_title(f"Original Spectrum\nMax: {spec_orig.max():.1f}")
    
    axs[1, 1].plot(spec_norm.numpy(), color='orange')
    axs[1, 1].set_title(f"Normalized Spectrum\nMax: {spec_norm.max():.1f} (Fixed to 1.0)")
    
    plt.tight_layout()
    plt.show()
    print("SpectralNormalization test done.")