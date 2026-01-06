# -*- coding: utf-8 -*-
"""
Trend Injection Augmentations.

Adds systematic drifts to the signal.
1. ExponentialTrend: Adds values following an exponential curve (e.g. overheating, capacitor charging).
2. LogarithmicTrend: Adds values following a log curve (e.g. saturation, cooling).

@author: Diyar Altinses, M.Sc.
"""

from typing import Tuple, Union, Optional
import torch


class ExponentialTrend(torch.nn.Module):
    """
    Adds an exponential drift to random segments of random channels.
    Formula: x + gain * (exp(decay * t) - 1)
    """

    def __init__(
        self, 
        probability: float = 1.0, 
        gain_range: Union[float, Tuple[float, float]] = (-1.0, 1.0),
        decay_range: Tuple[float, float] = (1.0, 3.0),
        length_range: Optional[Tuple[int, int]] = None,
        injection_prob: float = 0.5,
        trend_type: str = 'exponential'
    ):
        """
        Parameters
        ----------
        probability : float
            Global probability.
        gain_range : tuple
            Range for the magnitude of the trend.
        decay_range : tuple
            Steepness of the curve. Higher = sharper rise.
        length_range : tuple, optional
            Length of the trend segment.
        injection_prob : float
            Probability per channel to receive a trend.
        trend_type : str
            'exponential' or 'logarithmic'.
        """
        super().__init__()
        self.probability = probability
        self.gain_range = gain_range if isinstance(gain_range, (tuple, list)) else (gain_range, gain_range)
        self.decay_range = decay_range
        self.length_range = length_range
        self.injection_prob = injection_prob
        self.trend_type = trend_type

    def extra_repr(self) -> str:
        return f"prob={self.probability}, gain={self.gain_range}, type={self.trend_type}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input (N, C, L).
        """
        if torch.rand(1, device=x.device) > self.probability:
            return x

        N, C, L = x.shape
        device = x.device

        # 1. Active Channels Mask
        mask_c = (torch.rand(N, C, 1, device=device) < self.injection_prob).float()
        if mask_c.sum() == 0:
            return x

        # 2. Random Parameters
        gains = torch.empty(N, C, 1, device=device).uniform_(*self.gain_range)
        decays = torch.empty(N, C, 1, device=device).uniform_(*self.decay_range)
        
        # Lengths & Starts
        if self.length_range is None:
            lengths = torch.full((N, C, 1), L, device=device, dtype=torch.long)
            starts = torch.zeros((N, C, 1), device=device, dtype=torch.long)
        else:
            l_min, l_max = self.length_range
            l_max = min(l_max, L)
            lengths = torch.randint(l_min, l_max + 1, (N, C, 1), device=device)
            starts = (torch.rand(N, C, 1, device=device) * (L - lengths)).long()

        # 3. Time Grid & Segment Mask
        grid = torch.arange(L, device=device).view(1, 1, L)
        mask_t = (grid >= starts) & (grid < (starts + lengths))
        mask_t = mask_t.float()

        # 4. Generate Trend Curve
        # Normalized time t within the segment (0 to 1)
        # We need t to be 0 at 'starts' and 1 at 'starts + lengths'
        # t_local = (grid - starts) / lengths
        # Note: division by zero impossible since lengths >= 1
        t_local = (grid - starts) / lengths.float()
        
        # Mask out values outside segment to keep math clean (optional but good for debugging)
        t_local = t_local * mask_t

        if self.trend_type == 'exponential':
            # Formula: exp(decay * t) - 1
            # -1 ensures it starts at 0
            curve = torch.exp(decays * t_local) - 1.0
        elif self.trend_type == 'logarithmic':
            # Formula: log(decay * t + 1)
            curve = torch.log(decays * t_local + 1.0)
        else:
            curve = t_local # Fallback linear

        # 5. Apply Gain and Masks
        # Final shape: mask_c selects channels, mask_t selects time segment
        trend = gains * curve * mask_t * mask_c
        
        return x + trend


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # ---------------------------------------------------------
    # Test Data: Flat Signal
    # ---------------------------------------------------------
    L = 500
    data = torch.zeros(1, 2, L)
    
    # 1. Exponential Trend (Rising fast)
    aug_exp = ExponentialTrend(
        probability=1.0, 
        gain_range=(2.0, 3.0), 
        decay_range=(2.0, 5.0), # High decay = sharper curve
        injection_prob=1.0,
        trend_type='exponential'
    )
    out_exp = aug_exp(data.clone())

    # 2. Logarithmic Trend (Rising fast then slowing)
    aug_log = ExponentialTrend(
        probability=1.0, 
        gain_range=(2.0, 3.0), 
        decay_range=(5.0, 10.0), 
        injection_prob=1.0,
        trend_type='logarithmic'
    )
    out_log = aug_log(data.clone())

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------
    
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    
    axs[0].plot(out_exp[0, 0].numpy(), 'r', label="Exponential Trend")
    axs[0].set_title("Exponential Drift (e.g. Overheating)")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(out_log[0, 0].numpy(), 'g', label="Logarithmic Trend")
    axs[1].set_title("Logarithmic Drift (e.g. Saturation)")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    print("ExponentialTrend test done.")