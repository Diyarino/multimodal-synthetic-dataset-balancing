# -*- coding: utf-8 -*-
"""
Frequency Shift Augmentation.

Cyclically shifts the signal in the Frequency Domain.
Technically corresponds to Modulation in the Time/Spatial Domain.

@author: Diyar Altinses, M.Sc.
"""

import torch
import torch.fft


class FreqShift(torch.nn.Module):
    """
    Shifts the frequency spectrum of the input cyclically.
    
    Supports:
    - 1D Signals: (Batch, Channel, Length) -> Shifts frequency bins
    - 2D Images:  (Batch, Channel, Height, Width) -> Shifts 2D spectrum
    """

    def __init__(self, probability: float = 1.0, max_shift: int = 5):
        """
        Parameters
        ----------
        probability : float
            Probability of applying the augmentation.
        max_shift : int
            Maximum number of bins to shift the spectrum.
            Larger values = Stronger distortion.
        """
        super().__init__()
        self.probability = probability
        self.max_shift = max_shift

    def extra_repr(self) -> str:
        return f"probability={self.probability}, max_shift={self.max_shift}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor. 
                              Shape (N, C, L) for signals or (N, C, H, W) for images.
            
        Returns:
            torch.Tensor: Augmented tensor in Time/Spatial domain.
        """
        # 1. Probability Check
        if torch.rand(1, device=x.device) > self.probability:
            return x

        # 2. Determine FFT Dimensions
        # 3D Input (Signal) -> 1D FFT on last dim
        # 4D Input (Image)  -> 2D FFT on last two dims
        if x.ndim == 3:
            fft_dims = (-1,)
        elif x.ndim == 4:
            fft_dims = (-2, -1)
        else:
            return x # Unsupported shape

        # 3. FFT (Forward Transform)
        x_freq = torch.fft.fftn(x, dim=fft_dims)

        # 4. Apply Frequency Shift (Roll)
        # FIX: Generate independent shifts for EACH dimension in fft_dims.
        # This ensures len(shifts) == len(dims).
        
        # Create random shifts: One integer per dimension in fft_dims
        shifts = []
        for _ in fft_dims:
            s = torch.randint(-self.max_shift, self.max_shift + 1, (1,), device=x.device).item()
            shifts.append(s)
        
        # Convert list to tuple for torch.roll
        shifts = tuple(shifts)
        
        # Now: shifts is e.g. (3,) for 1D or (-2, 4) for 2D.
        # This matches fft_dims length perfectly.
        x_shifted = torch.roll(x_freq, shifts=shifts, dims=fft_dims)
        
        # 5. IFFT (Inverse Transform)
        x_restored = torch.fft.ifftn(x_shifted, dim=fft_dims)
        
        # 6. Restore Real & Type
        out = x_restored.real
        
        if x.dtype != out.dtype:
            out = out.to(x.dtype)

        return out


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # ---------------------------------------------------------
    # Test 1: 1D Signal (Sine Wave)
    # Shape: (1, 1, 100)
    # ---------------------------------------------------------
    t = torch.linspace(0, 1, 100)
    signal = torch.sin(2 * 3.14159 * 10 * t).view(1, 1, 100)
    
    aug_1d = FreqShift(probability=1.0, max_shift=10)
    out_1d = aug_1d(signal.clone())

    # ---------------------------------------------------------
    # Test 2: 2D Image
    # Shape: (1, 1, 100, 100)
    # ---------------------------------------------------------
    H, W = 100, 100
    img = torch.zeros(1, 1, H, W)
    y, x = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing='ij')
    # Create a pattern
    img[0, 0] = torch.sin(20 * x) * torch.cos(20 * y)
    
    aug_2d = FreqShift(probability=1.0, max_shift=5)
    out_2d = aug_2d(img.clone())

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    
    # 1D
    axs[0, 0].plot(signal[0, 0].numpy())
    axs[0, 0].set_title("Original 1D")
    
    axs[0, 1].plot(out_1d[0, 0].numpy(), color='orange')
    axs[0, 1].set_title("Freq Shifted 1D")
    
    # 2D
    axs[1, 0].imshow(img[0, 0], cmap='viridis')
    axs[1, 0].set_title("Original 2D")
    
    axs[1, 1].imshow(out_2d[0, 0], cmap='viridis')
    axs[1, 1].set_title("Freq Shifted 2D")
    
    plt.tight_layout()
    plt.show()
    print("FreqShift test done.")