# -*- coding: utf-8 -*-
"""
Spectral / Frequency Domain Augmentations.

Manipulates data in the Fourier Domain (Magnitude and Phase).
Supports both 1D (Time Series) and 2D (Images) inputs automatically.

@author: Diyar Altinses, M.Sc.
"""

from typing import Tuple
import torch
import torch.fft


class SpectralAugmentationBase(torch.nn.Module):
    """
    Base class for all Spectral Augmentations.
    Handles the common logic of FFT -> Modify -> IFFT.
    """
    def __init__(self, probability: float = 1.0):
        super().__init__()
        self.probability = probability

    def get_fft_dims(self, x: torch.Tensor) -> Tuple[int, ...]:
        """Auto-detects dimensions for FFT based on input shape."""
        # If 3D (N, C, L) -> 1D FFT on last dim
        if x.ndim == 3:
            return (-1,)
        # If 4D (N, C, H, W) -> 2D FFT on last two dims
        elif x.ndim == 4:
            return (-2, -1)
        else:
            raise ValueError(f"Input must be 3D (N,C,L) or 4D (N,C,H,W), got {x.ndim}D")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1, device=x.device) > self.probability:
            return x
            
        # 1. FFT
        dims = self.get_fft_dims(x)
        x_complex = torch.fft.fftn(x, dim=dims)
        
        # 2. Modify in Frequency Domain
        x_modified = self.modify_spectrum(x_complex, dims)
        
        # 3. IFFT
        # We take .real to discard tiny imaginary parts due to numerical errors
        x_restored = torch.fft.ifftn(x_modified, dim=dims).real
        
        # Restore dtype (if input was float32/float64)
        if x.dtype != x_restored.dtype:
            x_restored = x_restored.to(x.dtype)
            
        return x_restored

    def modify_spectrum(self, x_complex: torch.Tensor, dims: Tuple[int, ...]) -> torch.Tensor:
        """Override this method in subclasses."""
        raise NotImplementedError


class Fourier(SpectralAugmentationBase):
    """
    Applies FFT and immediately IFFT (Identity operation).
    Useful for testing the transformation pipeline or precision loss.
    """
    def modify_spectrum(self, x_complex: torch.Tensor, dims: Tuple[int, ...]) -> torch.Tensor:
        return x_complex  # Do nothing


class FreqShift(SpectralAugmentationBase):
    """
    Cyclically shifts the frequency spectrum.
    In the time domain, this corresponds to modulation / phase shifting.
    """
    def __init__(self, probability: float = 1.0, max_shift: int = 5):
        super().__init__(probability)
        self.max_shift = max_shift

    def modify_spectrum(self, x_complex: torch.Tensor, dims: Tuple[int, ...]) -> torch.Tensor:
        # Determine random shift
        shift = torch.randint(-self.max_shift, self.max_shift + 1, (1,), device=x_complex.device).item()
        
        # Apply roll along the frequency axes
        return torch.roll(x_complex, shifts=shift, dims=dims)
        
    def extra_repr(self) -> str:
        return f"probability={self.probability}, max_shift={self.max_shift}"


class FreqMasking(SpectralAugmentationBase):
    """
    Masks out random frequencies (sets magnitude to 0).
    Can simulate Low-Pass, High-Pass, or random Notch filters.
    """
    def __init__(self, probability: float = 1.0, mask_ratio: float = 0.1):
        """
        mask_ratio: Percentage of frequencies to zero out (0.0 to 1.0).
        """
        super().__init__(probability)
        self.mask_ratio = mask_ratio

    def modify_spectrum(self, x_complex: torch.Tensor, dims: Tuple[int, ...]) -> torch.Tensor:
        # Create a random binary mask
        # 1 = Keep, 0 = Drop
        # We generate random values and check if they are > mask_ratio
        mask = torch.rand_like(x_complex.real) > self.mask_ratio
        
        # Apply mask
        return x_complex * mask

    def extra_repr(self) -> str:
        return f"probability={self.probability}, mask_ratio={self.mask_ratio}"


class MagnitudeManipulation(SpectralAugmentationBase):
    """
    Randomly scales the magnitude of the spectrum components (Jitter).
    Preserves Phase.
    """
    def __init__(self, probability: float = 1.0, gain_range: Tuple[float, float] = (0.8, 1.2)):
        super().__init__(probability)
        self.gain_range = gain_range

    def modify_spectrum(self, x_complex: torch.Tensor, dims: Tuple[int, ...]) -> torch.Tensor:
        mag = torch.abs(x_complex)
        phase = torch.angle(x_complex)
        
        # Generate random gains for each frequency bin
        gains = torch.empty_like(mag).uniform_(self.gain_range[0], self.gain_range[1])
        
        # Apply Gain
        mag_modified = mag * gains
        
        # Reconstruct complex
        return torch.polar(mag_modified, phase)

    def extra_repr(self) -> str:
        return f"probability={self.probability}, gain_range={self.gain_range}"


class PhaseManipulation(SpectralAugmentationBase):
    """
    Randomly perturbs the phase of the spectrum.
    WARNING: Strong phase manipulation destroys signal structure heavily.
    """
    def __init__(self, probability: float = 1.0, strength: float = 0.1):
        """
        strength: Max random phase shift in radians (e.g. 0.1 means +/- 0.1 rad).
        """
        super().__init__(probability)
        self.strength = strength

    def modify_spectrum(self, x_complex: torch.Tensor, dims: Tuple[int, ...]) -> torch.Tensor:
        mag = torch.abs(x_complex)
        phase = torch.angle(x_complex)
        
        # Add random noise to phase
        noise = (torch.rand_like(phase) - 0.5) * 2 * self.strength
        phase_modified = phase + noise
        
        return torch.polar(mag, phase_modified)

    def extra_repr(self) -> str:
        return f"probability={self.probability}, strength={self.strength}"


class MagnitudeScaling(SpectralAugmentationBase):
    """
    Normalizes the Magnitude Spectrum to a specific range per sample.
    Essentially performs Min-Max normalization in the frequency domain.
    """
    def __init__(self, probability: float = 1.0, target_range: Tuple[float, float] = (0.0, 1.0)):
        super().__init__(probability)
        self.min_val, self.max_val = target_range

    def modify_spectrum(self, x_complex: torch.Tensor, dims: Tuple[int, ...]) -> torch.Tensor:
        mag = torch.abs(x_complex)
        phase = torch.angle(x_complex)
        
        # Find min/max per sample (N) and per channel (C)
        # We flatten the spatial/spectral dimensions for min/max calculation
        flat_dim_start = 2 # Dims 0(N) and 1(C) are preserved
        
        mag_flat = mag.flatten(start_dim=flat_dim_start)
        
        current_min = mag_flat.min(dim=-1, keepdim=True)[0]
        current_max = mag_flat.max(dim=-1, keepdim=True)[0]
        
        # Restore shapes for broadcasting
        # We need to unsqueeze the last dimensions to match 'mag'
        for _ in range(len(dims)):
            current_min = current_min.unsqueeze(-1)
            current_max = current_max.unsqueeze(-1)

        # Normalize to 0-1
        numerator = mag - current_min
        denominator = (current_max - current_min) + 1e-8
        mag_norm = numerator / denominator
        
        # Scale to target
        scale = self.max_val - self.min_val
        mag_scaled = mag_norm * scale + self.min_val
        
        return torch.polar(mag_scaled, phase)


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # ---------------------------------------------------------
    # Test 1: 1D Signal (e.g. Audio / Sensor Data)
    # Shape: (Batch=1, Channels=1, Length=100)
    # ---------------------------------------------------------
    t = torch.linspace(0, 10, 100)
    signal = (torch.sin(t) + 0.5 * torch.sin(3*t)).view(1, 1, 100)
    
    aug_freq_shift = FreqShift(probability=1.0, max_shift=10)
    aug_mask = FreqMasking(probability=1.0, mask_ratio=0.5)
    
    out_1d_shift = aug_freq_shift(signal.clone())
    out_1d_mask = aug_mask(signal.clone())

    # ---------------------------------------------------------
    # Test 2: 2D Signal (Image)
    # Shape: (Batch=1, Channels=1, Height=50, Width=50)
    # ---------------------------------------------------------
    img = torch.zeros(1, 1, 50, 50)
    y, x = torch.meshgrid(torch.linspace(0, 1, 50), torch.linspace(0, 1, 50), indexing='ij')
    img[0, 0] = torch.sin(10 * x) + torch.cos(10 * y) # Pattern
    
    aug_mag_jit = MagnitudeManipulation(probability=1.0, gain_range=(0.0, 2.0))
    out_2d = aug_mag_jit(img.clone())

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------[Image of Frequency Masking]

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    
    # Plot 1D Results
    axs[0, 0].plot(signal[0, 0].numpy(), label="Original")
    axs[0, 0].plot(out_1d_shift[0, 0].numpy(), label="Freq Shifted", alpha=0.7)
    axs[0, 0].set_title("1D Signal: Freq Shift")
    axs[0, 0].legend()
    
    axs[0, 1].plot(signal[0, 0].numpy(), label="Original")
    axs[0, 1].plot(out_1d_mask[0, 0].numpy(), label="Freq Masked", alpha=0.7)
    axs[0, 1].set_title("1D Signal: Freq Masking (Filtering)")
    axs[0, 1].legend()

    # Plot 2D Results
    axs[1, 0].imshow(img[0, 0], cmap='viridis')
    axs[1, 0].set_title("2D Original")
    
    axs[1, 1].imshow(out_2d[0, 0], cmap='viridis')
    axs[1, 1].set_title("2D Magnitude Jitter")
    
    plt.tight_layout()
    plt.show()
    print("Spectral Augmentation tests done.")