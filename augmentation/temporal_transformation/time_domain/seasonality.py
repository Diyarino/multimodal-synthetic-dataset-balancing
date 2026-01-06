# -*- coding: utf-8 -*-
"""
Seasonality / Sine Wave Injection.

Adds periodic signal components (sine waves) to the input.
Useful for simulating:
- Mains hum (50Hz/60Hz noise)
- Seasonal trends in forecasting
- Rotating machinery vibrations

@author: Diyar Altinses, M.Sc.
"""

from typing import Tuple, Union, Optional
import torch
import math


class AddSineWave(torch.nn.Module):
    """
    Adds a sine wave to random segments of random channels.
    Formula: x + gain * sin(2 * pi * freq * t + phase)
    """

    def __init__(
        self, 
        probability: float = 1.0, 
        gain_range: Union[float, Tuple[float, float]] = (0.1, 1.0), 
        freq_range: Union[float, Tuple[float, float]] = (1.0, 10.0),
        length_range: Optional[Tuple[int, int]] = None,
        injection_prob: float = 0.5
    ):
        """
        Parameters
        ----------
        probability : float
            Global probability of applying augmentation.
        gain_range : tuple
            Amplitude of the sine wave.
        freq_range : tuple
            Frequency of the sine wave (cycles per signal length).
            e.g. 5.0 means 5 full cycles fit into the signal.
        length_range : tuple, optional
            Length of the injected sine segment. If None, uses full signal length.
        injection_prob : float
            Probability per channel to receive a sine wave.
        """
        super().__init__()
        self.probability = probability
        
        # Standardize parameters to tuples
        self.gain_range = gain_range if isinstance(gain_range, (tuple, list)) else (gain_range, gain_range)
        self.freq_range = freq_range if isinstance(freq_range, (tuple, list)) else (freq_range, freq_range)
        self.length_range = length_range 
        self.injection_prob = injection_prob

    def extra_repr(self) -> str:
        return f"prob={self.probability}, gain={self.gain_range}, freq={self.freq_range}, inject_p={self.injection_prob}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input (N, C, L).
        """
        if torch.rand(1, device=x.device) > self.probability:
            return x

        N, C, L = x.shape
        device = x.device

        # 1. Determine active channels
        # Mask: (N, C, 1)
        mask_c = (torch.rand(N, C, 1, device=device) < self.injection_prob).float()
        
        if mask_c.sum() == 0:
            return x

        # 2. Generate Random Parameters (Vectorized)
        
        # Gain & Frequency: (N, C, 1)
        gains = torch.empty(N, C, 1, device=device).uniform_(*self.gain_range)
        freqs = torch.empty(N, C, 1, device=device).uniform_(*self.freq_range)
        
        # Lengths & Starts
        if self.length_range is None:
            # Full length
            lengths = torch.full((N, C, 1), L, device=device, dtype=torch.long)
            starts = torch.zeros((N, C, 1), device=device, dtype=torch.long)
        else:
            # Random length
            l_min, l_max = self.length_range
            # Clamp max length to L
            l_max = min(l_max, L)
            lengths = torch.randint(l_min, l_max + 1, (N, C, 1), device=device)
            
            # Random start
            max_starts = L - lengths
            starts = (torch.rand(N, C, 1, device=device) * max_starts).long()

        # 3. Create Time Grid & Mask
        # Grid: (1, 1, L) -> [0, 1, ..., L-1]
        grid = torch.arange(L, device=device).view(1, 1, L)
        
        # Time Mask: True inside the segment
        mask_t = (grid >= starts) & (grid < (starts + lengths))
        mask_t = mask_t.float()

        # 4. Generate Sine Wave
        # To generate correct sine waves relative to the segment start or absolute time:
        # Here we use absolute time [0, 1] normalized over L.
        t_norm = torch.linspace(0, 1, L, device=device).view(1, 1, L)
        
        # Wave: sin(2 * pi * freq * t)
        # Broadcasting: (N,C,1) * (1,1,L) -> (N,C,L)
        sine_wave = torch.sin(2 * math.pi * freqs * t_norm)
        
        # Scale by gain
        sine_wave = sine_wave * gains
        
        # 5. Combine Masks and Add
        # Only add sine where both channel is selected AND time is within segment
        final_mask = mask_c * mask_t
        
        return x + (sine_wave * final_mask)


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # ---------------------------------------------------------
    # Test Data: Linear Trend
    # ---------------------------------------------------------
    L = 500
    t = torch.linspace(0, 10, L)
    # 3 Channels
    data = t.view(1, 1, L).repeat(1, 3, 1)
    
    # ---------------------------------------------------------
    # Augmentation
    # ---------------------------------------------------------
    # Add high frequency noise (freq 20-50) to short segments (50-150)
    aug = AddSineWave(
        probability=1.0, 
        gain_range=(0.5, 1.0), 
        freq_range=(20.0, 50.0), 
        length_range=(50, 150), 
        injection_prob=0.8
    )
    
    out = aug(data.clone())

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    colors = ['blue', 'orange', 'green']
    
    for c in range(3):
        axs[c].plot(data[0, c].numpy(), 'k--', alpha=0.5, label="Original (Trend)")
        axs[c].plot(out[0, c].numpy(), color=colors[c], label="Augmented")
        axs[c].set_title(f"Channel {c}")
        axs[c].legend(loc='upper right')
        axs[c].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    print("Seasonality test done.")