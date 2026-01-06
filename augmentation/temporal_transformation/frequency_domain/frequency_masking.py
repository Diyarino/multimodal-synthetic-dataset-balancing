# -*- coding: utf-8 -*-
"""
Frequency Masking Augmentation.

Masks out specific frequency components in the Fourier domain.
Can simulate low-pass filtering, high-pass filtering, or random spectral dropout.

@author: Diyar Altinses, M.Sc.
"""

from typing import Tuple
import torch
import torch.fft

class FreqMasking(torch.nn.Module):
    """
    Randomly masks out frequencies in the spectral domain.
    
    Modes:
    - 'random': Randomly drops individual frequency bins (Spectral Dropout).
    - 'lowpass': Keeps only low frequencies (Blur).
    - 'highpass': Keeps only high frequencies (Edge detection).
    """

    def __init__(
        self, 
        probability: float = 1.0, 
        mask_ratio: float = 0.1, 
        mode: str = 'random'
    ):
        """
        Parameters
        ----------
        probability : float
            Probability of applying the augmentation.
        mask_ratio : float
            Percentage of the spectrum to mask out (0.0 to 1.0).
            For 'lowpass'/'highpass', this defines the cutoff frequency.
        mode : str
            'random', 'lowpass', or 'highpass'.
        """
        super().__init__()
        self.probability = probability
        self.mask_ratio = mask_ratio
        self.mode = mode.lower()

    def extra_repr(self) -> str:
        return f"probability={self.probability}, mask_ratio={self.mask_ratio}, mode={self.mode}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor (N, C, L) or (N, C, H, W).
            
        Returns:
            torch.Tensor: Augmented tensor.
        """
        # 1. Probability Check
        if torch.rand(1, device=x.device) > self.probability:
            return x

        # 2. Determine FFT Dimensions
        # 3D Input (Signal) -> 1D FFT on last dim
        # 4D Input (Image)  -> 2D FFT on last two dims
        fft_dims = (-1,) if x.ndim == 3 else (-2, -1)

        # 3. FFT
        # Use rfft (Real FFT) for efficiency since input is real.
        # It returns only the positive half of the spectrum.
        x_freq = torch.fft.rfftn(x, dim=fft_dims)
        
        # 4. Create Mask
        mask = self._create_mask(x_freq, fft_dims)
        
        # 5. Apply Mask
        x_masked = x_freq * mask

        # 6. Inverse FFT
        # irfftn automatically handles the complex-to-real conversion correctly
        x_restored = torch.fft.irfftn(x_masked, s=x.shape[fft_dims[0]:], dim=fft_dims)

        return x_restored

    def _create_mask(self, x_freq: torch.Tensor, dims: Tuple[int, ...]) -> torch.Tensor:
        """Generates the frequency mask based on the selected mode."""
        shape = x_freq.shape
        device = x_freq.device
        
        if self.mode == 'random':
            # Random Dropout: Create a binary mask where 1 = Keep, 0 = Drop
            # torch.rand gives [0, 1]. Keep if rand > mask_ratio.
            return (torch.rand(shape, device=device) > self.mask_ratio).float()

        elif self.mode == 'lowpass':
            # Keep low frequencies (start of the spectrum), mask high ones (end).
            # For simplicity in N-D, we just mask the last part of the last dimension
            # (which usually contains the highest frequencies in rfft).
            last_dim_size = shape[-1]
            cutoff = int(last_dim_size * (1 - self.mask_ratio))
            
            mask = torch.ones(shape, device=device)
            mask[..., cutoff:] = 0.0
            return mask

        elif self.mode == 'highpass':
            # Keep high frequencies, mask low ones (start of spectrum).
            # Note: Masking DC component (index 0) removes the mean/brightness!
            last_dim_size = shape[-1]
            cutoff = int(last_dim_size * self.mask_ratio)
            
            mask = torch.ones(shape, device=device)
            mask[..., :cutoff] = 0.0
            return mask
        
        else:
            return torch.ones(shape, device=device)

# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # ---------------------------------------------------------
    # Test 1: 1D Signal (Sine Wave Mix)
    # ---------------------------------------------------------
    t = torch.linspace(0, 1, 100)
    # Low freq (5Hz) + High freq (30Hz)
    signal = torch.sin(2*3.14*5*t) + 0.5*torch.sin(2*3.14*30*t)
    signal = signal.view(1, 1, -1)

    # Lowpass Filter: Should remove the 30Hz jitter and keep the 5Hz wave
    aug_lp = FreqMasking(probability=1.0, mask_ratio=0.7, mode='lowpass')
    out_lp = aug_lp(signal)

    # ---------------------------------------------------------
    # Test 2: 2D Image
    # ---------------------------------------------------------
    H, W = 100, 100
    img = torch.zeros(1, 1, H, W)
    y, x = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing='ij')
    img[0, 0] = torch.sin(10*x) + torch.cos(10*y) # Pattern

    # Random Masking (Spectral Dropout)
    aug_rand = FreqMasking(probability=1.0, mask_ratio=0.8, mode='random')
    out_rand = aug_rand(img)

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    
    # 1D
    axs[0, 0].plot(signal[0, 0].numpy())
    axs[0, 0].set_title("Original Signal (Low + High Freq)")
    
    axs[0, 1].plot(out_lp[0, 0].numpy(), color='orange')
    axs[0, 1].set_title("LowPass Filtered (High Freq removed)")
    
    # 2D
    axs[1, 0].imshow(img[0, 0], cmap='viridis')
    axs[1, 0].set_title("Original Image")
    
    axs[1, 1].imshow(out_rand[0, 0], cmap='viridis')
    axs[1, 1].set_title("Spectral Dropout (Random Masking)")
    
    plt.tight_layout()
    plt.show()
    print("FreqMasking test done.")