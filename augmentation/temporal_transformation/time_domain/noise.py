# -*- coding: utf-8 -*-
"""
Gaussian Noise Augmentations.

1. AdditiveGaussianNoise: Standard additive white gaussian noise (Global).
2. LocalGaussianNoise: Adds noise only to specific time segments and channels.

@author: Diyar Altinses, M.Sc.
"""

from typing import Tuple, Union, Optional
import torch


class AdditiveGaussianNoise(torch.nn.Module):
    """
    Adds Gaussian Noise to the entire tensor.
    Formula: x = x + N(mean, std)
    """

    def __init__(
        self, 
        probability: float = 1.0, 
        mean: float = 0.0, 
        std: float = 1.0, 
        clamp_range: Optional[Tuple[float, float]] = None
    ):
        """
        Parameters
        ----------
        probability : float
            Probability of applying the augmentation.
        mean : float
            Mean of the Gaussian distribution.
        std : float
            Standard deviation (noise strength).
        clamp_range : tuple, optional
            (min, max) to clamp the result (e.g., (-1, 1) or (0, 1)).
        """
        super().__init__()
        self.probability = probability
        self.mean = mean
        self.std = std
        self.clamp_range = clamp_range

    def extra_repr(self) -> str:
        return f"prob={self.probability}, mean={self.mean}, std={self.std}, clamp={self.clamp_range}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Noisy tensor.
        """
        if torch.rand(1, device=x.device) > self.probability:
            return x

        # Generate Noise
        noise = torch.randn_like(x) * self.std + self.mean
        out = x + noise

        # Optional Clamping
        if self.clamp_range is not None:
            out = torch.clamp(out, min=self.clamp_range[0], max=self.clamp_range[1])
            
        return out


class LocalGaussianNoise(torch.nn.Module):
    """
    Adds Gaussian Noise to RANDOM segments of RANDOM channels.
    Simulates sensor-specific interference or short bursts of noise.
    
    (Formerly Gausnoise_perCH).
    """

    def __init__(
        self, 
        probability: float = 1.0, 
        mean_range: Union[float, Tuple[float, float]] = (0.0, 0.0), 
        std_range: Union[float, Tuple[float, float]] = (0.1, 0.5),
        length_range: Optional[Tuple[int, int]] = None,
        injection_prob: float = 0.5
    ):
        """
        Parameters
        ----------
        probability : float
            Global probability of applying augmentation.
        mean_range : float or tuple
            Range to pick the noise mean from.
        std_range : float or tuple
            Range to pick the noise std from.
        length_range : tuple (min, max)
            Length of the noise segment. If None, defaults to (10, 50).
        injection_prob : float
            Probability per channel to receive noise.
        """
        super().__init__()
        self.probability = probability
        
        # Standardize ranges to tuples
        self.mean_range = mean_range if isinstance(mean_range, (tuple, list)) else (mean_range, mean_range)
        self.std_range = std_range if isinstance(std_range, (tuple, list)) else (std_range, std_range)
        self.length_range = length_range if length_range is not None else (10, 50)
        self.injection_prob = injection_prob

    def extra_repr(self) -> str:
        return f"prob={self.probability}, std_range={self.std_range}, len_range={self.length_range}, inject_p={self.injection_prob}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input (N, C, L).
        """
        if torch.rand(1, device=x.device) > self.probability:
            return x

        N, C, L = x.shape
        device = x.device

        # 1. Determine active channels (Vectorized)
        # mask_c: (N, C, 1) -> 1.0 if channel gets noise, 0.0 otherwise
        mask_c = (torch.rand(N, C, 1, device=device) < self.injection_prob).float()
        
        if mask_c.sum() == 0:
            return x

        # 2. Generate Random Parameters per Channel (Vectorized)
        
        # Random Mean & Std
        # Shape: (N, C, 1)
        means = torch.empty(N, C, 1, device=device).uniform_(*self.mean_range)
        stds = torch.empty(N, C, 1, device=device).uniform_(*self.std_range)
        
        # Random Start & Length
        # Lengths clamped to max signal length
        l_min, l_max = self.length_range
        lengths = torch.randint(l_min, min(l_max, L) + 1, (N, C, 1), device=device)
        
        # Start positions (must fit the length)
        # Since lengths vary, we calculate max_start per channel
        # max_start = L - length
        max_starts = L - lengths
        # Generate random start <= max_start
        rand_factors = torch.rand(N, C, 1, device=device)
        starts = (rand_factors * max_starts).long()

        # 3. Create Time Mask (Vectorized Broadcasting)
        # Grid: (1, 1, L) -> [0, 1, 2, ... L-1]
        grid = torch.arange(L, device=device).view(1, 1, L)
        
        # mask_t is True where: start <= grid < start + length
        # Broadcasting: (N, C, 1) vs (1, 1, L) -> (N, C, L)
        mask_t = (grid >= starts) & (grid < (starts + lengths))
        mask_t = mask_t.float()

        # 4. Generate Noise
        # Base noise for whole tensor
        noise = torch.randn_like(x) * stds + means
        
        # 5. Combine Masks and Apply
        # Final Mask = Channel Mask * Time Mask
        # If mask is 1, add noise. If 0, add 0.
        final_mask = mask_c * mask_t
        
        return x + (noise * final_mask)


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # ---------------------------------------------------------
    # Test Data: 3 Channels with different offsets
    # ---------------------------------------------------------
    # Channel 0: Straight line at 0
    # Channel 1: Straight line at 5
    # Channel 2: Straight line at 10
    L = 1000
    data = torch.zeros(1, 3, L)
    data[0, 0] = 0
    data[0, 1] = 5
    data[0, 2] = 10
    
    # ---------------------------------------------------------
    # Apply Local Noise
    # ---------------------------------------------------------
    # High injection prob (0.8) -> Most channels should be hit
    # Length (100, 300) -> Short bursts
    aug = LocalGaussianNoise(
        probability=1.0, 
        mean_range=0.0, 
        std_range=(0.2, 1.0), 
        length_range=(100, 300), 
        injection_prob=0.8
    )
    
    out = aug(data.clone())

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    colors = ['blue', 'orange', 'green']
    
    for c in range(3):
        axs[c].plot(data[0, c].numpy(), 'k--', alpha=0.5, label="Original")
        axs[c].plot(out[0, c].numpy(), color=colors[c], label="Augmented")
        axs[c].set_title(f"Channel {c}")
        axs[c].legend(loc='upper right')
        axs[c].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    print("Gaussian Noise tests done.")