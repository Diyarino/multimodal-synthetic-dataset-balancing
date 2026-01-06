# -*- coding: utf-8 -*-
"""
Spectral Phase Shift Augmentation.

Applies a constant shift to the phase spectrum.
In the Time Domain, this results in a cyclic translation (Time Shift) or structural rotation.
This is mathematically cleaner than 'scaling' the phase.

@author: Diyar Altinses, M.Sc.
"""

from typing import Tuple
import torch
import math


class SpectralPhaseShift(torch.nn.Module):
    """
    Shifts the phase of the signal by a random amount.
    
    Effect:
    - Global Shift: Translates the signal in time/space (Cyclic Shift).
    - Random Shift: Randomizes structure (strong distortion).
    """

    def __init__(
        self, 
        probability: float = 1.0, 
        shift_range: Tuple[float, float] = (-0.5, 0.5),
        mode: str = 'global'
    ):
        """
        Parameters
        ----------
        probability : float
            Probability of applying the augmentation.
        shift_range : tuple
            Range of phase shift in radians * PI.
            Example: (-0.5, 0.5) means shifting by -0.5*PI to +0.5*PI.
        mode : str
            'global': Adds same random shift to ALL frequencies (Preserves relative phase).
                      This creates a pure time-shift effect.
            'random': Adds different random shift to EACH frequency bin.
                      This scrambles the signal structure (Distortion).
        """
        super().__init__()
        self.probability = probability
        self.shift_range = shift_range
        self.mode = mode

    def extra_repr(self) -> str:
        return f"probability={self.probability}, shift_range={self.shift_range}pi, mode={self.mode}"

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

        # 2. Determine Dimensions
        if x.ndim == 3:
            fft_dims = (-1,)
        elif x.ndim == 4:
            fft_dims = (-2, -1)
        else:
            return x

        # 3. FFT
        x_freq = torch.fft.rfftn(x, dim=fft_dims)
        
        mag = torch.abs(x_freq)
        phase = torch.angle(x_freq)

        # 4. Generate Phase Shift
        # We shift by value * PI
        low, high = self.shift_range
        
        if self.mode == 'global':
            # One random value per sample in batch
            # Shape: (N, 1, 1...) to broadcast
            shifts = torch.empty(x.shape[0], device=x.device).uniform_(low, high)
            
            # Reshape for broadcasting
            # We need (N, 1, 1) for 1D or (N, 1, 1, 1) for 2D
            view_shape = [x.shape[0]] + [1] * (phase.ndim - 1)
            shifts = shifts.view(*view_shape) * math.pi
            
        else:
            # Random shift per frequency bin
            shifts = torch.empty_like(phase).uniform_(low, high) * math.pi

        # 5. Apply Shift
        # New Phase = Old Phase + Shift
        phase_shifted = phase + shifts

        # 6. Reconstruct
        x_modified = torch.polar(mag, phase_shifted)
        out = torch.fft.irfftn(x_modified, s=x.shape[fft_dims[0]:], dim=fft_dims)

        return out


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # ---------------------------------------------------------
    # Test: 1D Pulse
    # ---------------------------------------------------------
    L = 100
    signal = torch.zeros(1, 1, L)
    # Create a distinct shape (Triangle)
    signal[0, 0, 40:60] = torch.tensor([
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 
        1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1
    ])
    
    # Mode 1: Global Shift (Should move the triangle cyclically)
    aug_global = SpectralPhaseShift(probability=1.0, shift_range=(0.2, 0.2), mode='global')
    out_global = aug_global(signal.clone())

    # Mode 2: Random Shift (Should destroy the triangle shape)
    aug_rand = SpectralPhaseShift(probability=1.0, shift_range=(-1.0, 1.0), mode='random')
    out_rand = aug_rand(signal.clone())

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------
    

    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    
    axs[0].plot(signal[0, 0].numpy(), color='black')
    axs[0].set_title("Original Signal (Triangle)")
    axs[0].grid(True, alpha=0.3)
    
    axs[1].plot(out_global[0, 0].numpy(), color='blue')
    axs[1].set_title("Global Phase Shift (Cyclic Translation/Distortion)")
    axs[1].grid(True, alpha=0.3)
    
    axs[2].plot(out_rand[0, 0].numpy(), color='red')
    axs[2].set_title("Random Phase Shift (Structure Scrambled)")
    axs[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    print("SpectralPhaseShift test done.")