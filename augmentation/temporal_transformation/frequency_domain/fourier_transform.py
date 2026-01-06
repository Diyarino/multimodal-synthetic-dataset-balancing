# -*- coding: utf-8 -*-
"""
Fourier Transform Module.

Converts signals or images from the Time/Spatial Domain to the Frequency Domain.
Supports automatic 1D/2D detection.

@author: Diyar Altinses, M.Sc.
"""

import torch
import torch.fft


class FourierTransform(torch.nn.Module):
    """
    Applies the Fast Fourier Transform (FFT) to the input.
    
    - For 3D inputs (Batch, Channels, Length), applies 1D FFT.
    - For 4D inputs (Batch, Channels, Height, Width), applies 2D FFT.
    
    Can return either raw Complex tensors or Magnitude Spectrums.
    """

    def __init__(self, probability: float = 1.0, return_magnitude: bool = False):
        """
        Parameters
        ----------
        probability : float
            Probability of applying the transform.
        return_magnitude : bool
            If True, returns the absolute value (magnitude spectrum).
            If False, returns the complex tensor (real + imag). 
            Default is False (matches original behavior).
        """
        super().__init__()
        self.probability = probability
        self.return_magnitude = return_magnitude

    def extra_repr(self) -> str:
        return f"probability={self.probability}, return_magnitude={self.return_magnitude}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor. 
                              Shape (N, C, L) for signals or (N, C, H, W) for images.
            
        Returns:
            torch.Tensor: Frequency domain representation.
                          Note: If return_magnitude=False, output is Complex64/128!
        """
        # 1. Probability Check
        if torch.rand(1, device=x.device) > self.probability:
            return x

        # 2. Determine Dimensions
        if x.ndim == 3:
            # 1D Signal: (N, C, L) -> FFT on last dim
            # rfft is often better for real inputs (returns only half spectrum), 
            # but we stick to standard fft for consistency.
            out = torch.fft.fft(x, dim=-1)
            
        elif x.ndim == 4:
            # 2D Image: (N, C, H, W) -> FFT on last two dims
            out = torch.fft.fft2(x, dim=(-2, -1))
            
            # Optional: Shift zero frequency to center (better for visualization)
            # out = torch.fft.fftshift(out, dim=(-2, -1))
            
        else:
            # Fallback for other shapes
            out = torch.fft.fft(x)

        # 3. Return Mode
        if self.return_magnitude:
            # Returns real-valued tensor (Magnitude Spectrum)
            # Add small epsilon to avoid log(0) if you plan to log-scale later
            return torch.abs(out)
        else:
            # Returns complex-valued tensor
            return out


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # ---------------------------------------------------------
    # Test 1: 1D Signal (Sine Wave)
    # ---------------------------------------------------------
    # Create a 10 Hz sine wave
    fs = 100 # Sampling rate
    t = torch.linspace(0, 1, fs)
    # Signal: 10 Hz + 20 Hz
    signal = (torch.sin(2 * 3.14159 * 10 * t) + 0.5 * torch.sin(2 * 3.14159 * 20 * t))
    signal = signal.view(1, 1, -1) # (N, C, L)
    
    # Transform (Magnitude)
    aug_1d = FourierTransform(probability=1.0, return_magnitude=True)
    out_1d = aug_1d(signal)

    # ---------------------------------------------------------
    # Test 2: 2D Image (Patterns)
    # ---------------------------------------------------------
    H, W = 100, 100
    img = torch.zeros(1, 1, H, W)
    y, x = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing='ij')
    # Vertical stripes pattern
    img[0, 0] = torch.sin(20 * x)
    
    # Transform (Complex -> we show magnitude manually)
    aug_2d = FourierTransform(probability=1.0, return_magnitude=False)
    out_2d_complex = aug_2d(img)
    
    # Shift center for visualization
    out_2d_mag = torch.fft.fftshift(torch.abs(out_2d_complex), dim=(-2, -1))

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------
    

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    
    # 1D Plots
    axs[0, 0].plot(signal[0, 0].numpy())
    axs[0, 0].set_title("1D Original (Time Domain)")
    
    # Plot only first half of spectrum (positive frequencies)
    half_point = out_1d.shape[-1] // 2
    axs[0, 1].plot(out_1d[0, 0, :half_point].numpy())
    axs[0, 1].set_title("1D FFT (Magnitude Spectrum)\nPeaks at 10Hz and 20Hz")
    axs[0, 1].grid(True)
    
    # 2D Plots
    axs[1, 0].imshow(img[0, 0], cmap='gray')
    axs[1, 0].set_title("2D Original (Vertical Stripes)")
    
    # Log scale for better visibility of spectrum
    axs[1, 1].imshow(torch.log(out_2d_mag[0, 0] + 1), cmap='inferno')
    axs[1, 1].set_title("2D FFT (Centered Spectrum)\nDots show frequency components")
    
    plt.tight_layout()
    plt.show()
    print("FourierTransform test done.")