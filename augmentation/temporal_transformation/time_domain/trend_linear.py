# -*- coding: utf-8 -*-
"""
Linear Trend Injection.

Adds a linear drift (slope + bias) to the signal.
Common in sensors due to temperature changes or calibration drift.

@author: Diyar Altinses, M.Sc.
"""

from typing import Tuple, Union, Optional
import torch

class LinearTrend(torch.nn.Module):
    """
    Adds a linear trend to random segments of random channels.
    Formula: x + (slope * t + bias)
    """

    def __init__(
        self, 
        probability: float = 1.0, 
        slope_range: Union[float, Tuple[float, float]] = (-0.01, 0.01),
        bias_range: Union[float, Tuple[float, float]] = (0.0, 0.0),
        length_range: Optional[Tuple[int, int]] = None,
        injection_prob: float = 0.5
    ):
        """
        Parameters
        ----------
        probability : float
            Global probability of applying augmentation.
        slope_range : tuple
            Range for the slope of the linear trend (per time step).
        bias_range : tuple
            Range for the bias (offset) added to the trend.
        length_range : tuple, optional
            Length of the trend segment. If None, uses full signal length.
        injection_prob : float
            Probability per channel to receive a trend.
        """
        super().__init__()
        self.probability = probability
        
        # Standardize ranges to tuples
        self.slope_range = slope_range if isinstance(slope_range, (tuple, list)) else (slope_range, slope_range)
        self.bias_range = bias_range if isinstance(bias_range, (tuple, list)) else (bias_range, bias_range)
        self.length_range = length_range 
        self.injection_prob = injection_prob

    def extra_repr(self) -> str:
        return f"prob={self.probability}, slope={self.slope_range}, bias={self.bias_range}, inject_p={self.injection_prob}"

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
        
        # Slope & Bias: (N, C, 1)
        slopes = torch.empty(N, C, 1, device=device).uniform_(*self.slope_range)
        biases = torch.empty(N, C, 1, device=device).uniform_(*self.bias_range)
        
        # Lengths & Starts
        if self.length_range is None:
            # Full length trend
            lengths = torch.full((N, C, 1), L, device=device, dtype=torch.long)
            starts = torch.zeros((N, C, 1), device=device, dtype=torch.long)
        else:
            # Random segment trend
            l_min, l_max = self.length_range
            l_max = min(l_max, L)
            lengths = torch.randint(l_min, l_max + 1, (N, C, 1), device=device)
            
            # Random start positions
            max_starts = L - lengths
            starts = (torch.rand(N, C, 1, device=device) * max_starts).long()

        # 3. Create Time Grid & Mask
        # Grid: (1, 1, L) -> [0, 1, ..., L-1]
        grid = torch.arange(L, device=device).view(1, 1, L)
        
        # Time Mask: True inside the segment
        mask_t = (grid >= starts) & (grid < (starts + lengths))
        mask_t = mask_t.float()

        # 4. Generate Linear Trend
        # We define t=0 at the start of the segment
        # t_local = (grid - starts)
        t_local = (grid - starts).float()
        
        # Apply formula: slope * t + bias
        # We only care about values inside mask_t, so we can multiply t_local by mask_t
        t_local = t_local * mask_t
        
        trend = (slopes * t_local) + biases
        
        # 5. Apply Trend
        # Only add trend where channel is selected AND time is within segment
        # Note: 'trend' already zeroed out outside time segment via t_local logic above? 
        # Actually bias is constant, so we need to mask the whole expression.
        
        final_mask = mask_c * mask_t
        
        return x + (trend * final_mask)


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # ---------------------------------------------------------
    # Test Data: Random Noise around 0
    # ---------------------------------------------------------
    L = 1000
    data = torch.randn(1, 3, L) * 0.1
    
    # ---------------------------------------------------------
    # Augmentation: Add strong drift
    # ---------------------------------------------------------
    # Slope: +/- 0.01 per step (over 500 steps -> +/- 5.0 total change)
    aug = LinearTrend(
        probability=1.0, 
        slope_range=(-0.01, 0.01), 
        bias_range=(0.0, 1.0), 
        length_range=(200, 500), 
        injection_prob=1.0 # Hit all channels
    )
    
    out = aug(data.clone())

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    colors = ['blue', 'orange', 'green']
    
    for c in range(3):
        axs[c].plot(data[0, c].numpy(), 'k', alpha=0.3, label="Original (Noise)")
        axs[c].plot(out[0, c].numpy(), color=colors[c], label="Augmented (Drift)")
        axs[c].set_title(f"Channel {c}: Linear Drift Injection")
        axs[c].legend(loc='upper right')
        axs[c].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    print("LinearTrend test done.")