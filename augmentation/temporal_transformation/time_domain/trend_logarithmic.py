# -*- coding: utf-8 -*-
"""
Logarithmic Trend Injection.

Adds a logarithmic drift to the signal.
Typical for saturation processes (e.g. capacitor charging, sensor warming up then stabilizing).

@author: Diyar Altinses, M.Sc.
"""

from typing import Tuple, Union, Optional
import torch
import torch.nn as nn


class LogTrend(nn.Module):
    """
    Adds a logarithmic trend to random segments of random channels.
    Formula: x + gain * log(t + 1)
    """

    def __init__(
        self, 
        probability: float = 1.0, 
        gain_range: Union[float, Tuple[float, float]] = (-1.0, 1.0),
        length_range: Optional[Tuple[int, int]] = None,
        injection_prob: float = 0.5
    ):
        """
        Parameters
        ----------
        probability : float
            Global probability of applying augmentation.
        gain_range : tuple
            Range for the magnitude/direction of the trend.
        length_range : tuple, optional
            Length of the trend segment. If None, uses full signal length.
        injection_prob : float
            Probability per channel to receive a trend.
        """
        super().__init__()
        self.probability = probability
        
        # Standardize ranges
        self.gain_range = gain_range if isinstance(gain_range, (tuple, list)) else (gain_range, gain_range)
        self.length_range = length_range 
        self.injection_prob = injection_prob

    def extra_repr(self) -> str:
        return f"prob={self.probability}, gain={self.gain_range}, inject_p={self.injection_prob}"

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
        mask_c = (torch.rand(N, C, 1, device=device) < self.injection_prob).float()
        
        if mask_c.sum() == 0:
            return x

        # 2. Generate Random Parameters
        gains = torch.empty(N, C, 1, device=device).uniform_(*self.gain_range)
        
        # Lengths & Starts
        if self.length_range is None:
            lengths = torch.full((N, C, 1), L, device=device, dtype=torch.long)
            starts = torch.zeros((N, C, 1), device=device, dtype=torch.long)
        else:
            l_min, l_max = self.length_range
            l_max = min(l_max, L)
            lengths = torch.randint(l_min, l_max + 1, (N, C, 1), device=device)
            max_starts = L - lengths
            starts = (torch.rand(N, C, 1, device=device) * max_starts).long()

        # 3. Create Time Grid & Segment Mask
        grid = torch.arange(L, device=device).view(1, 1, L)
        
        # Mask is True where: start <= t < start + length
        mask_t = (grid >= starts) & (grid < (starts + lengths))
        mask_t = mask_t.float()

        # 4. Generate Log Curve
        # Local time t starting at 0 within the segment
        t_local = (grid - starts).float()
        
        # Normalize time roughly to range [0, 10] or similar to make log distinctive?
        # Standard log(t) grows very slowly. 
        # Often useful to scale t relative to length: log( (t/len)*scale + 1 )
        # Here we implement simple log(t + 1) to match your original intent but safely.
        
        # Only compute log where t >= 0 (handled by mask later, but for safety adding 1)
        # log(0+1) = 0. log(10+1) ~ 2.4. 
        curve = torch.log(t_local + 1.0)
        
        # 5. Apply
        # Apply mask_t to zero out curve outside segment
        trend = gains * curve * mask_t * mask_c
        
        return x + trend


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # ---------------------------------------------------------
    # Test Data: Constant Baseline
    # ---------------------------------------------------------
    L = 500
    data = torch.zeros(1, 1, L)
    
    # ---------------------------------------------------------
    # Augmentation
    # ---------------------------------------------------------
    # Add strong positive log trend
    aug = LogTrend(
        probability=1.0, 
        gain_range=(2.0, 5.0), 
        length_range=None, # Full length 
        injection_prob=1.0
    )
    
    out = aug(data.clone())

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    ax.plot(out[0, 0].numpy(), 'g-', label="Logarithmic Trend")
    ax.set_title("LogTrend Injection (Saturation Curve)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    print("LogTrend test done.")