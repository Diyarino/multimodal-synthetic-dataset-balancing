# -*- coding: utf-8 -*-
"""
Channel-wise Scaling and Clamping.

1. RandomChannelScaling: Rescales specific channels to a new range (Min-Max normalization subset).
2. RandomClamp: Clips values of specific channels to a min/max limit.

@author: Diyar Altinses, M.Sc.
"""

import torch


class RandomChannelScaling(torch.nn.Module):
    """
    Randomly selects channels and rescales them to a target range [min, max].
    Unlike global normalization, this creates variance between channels.
    
    Formula: X_new = (X - X.min) / (X.max - X.min) * (target_max - target_min) + target_min
    """

    def __init__(
        self, 
        probability: float = 1.0, 
        min_val: float = 0.0, 
        max_val: float = 1.0, 
        channel_prob: float = 0.5
    ):
        """
        Parameters
        ----------
        probability : float
            Global probability of applying the augmentation.
        min_val, max_val : float
            Target range for the scaled channels.
        channel_prob : float
            Probability for each channel to be selected for scaling (0.0 to 1.0).
        """
        super().__init__()
        self.probability = probability
        self.min_val = min_val
        self.max_val = max_val
        self.channel_prob = channel_prob

    def extra_repr(self) -> str:
        return f"prob={self.probability}, range=[{self.min_val}, {self.max_val}], ch_prob={self.channel_prob}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input (N, C, L).
        """
        if torch.rand(1, device=x.device) > self.probability:
            return x

        N, C = x.shape[:2]
        device = x.device

        # 1. Determine active channels
        # Mask: (N, C, 1) -> 1.0 if channel should be scaled, 0.0 otherwise
        mask = (torch.rand(N, C, 1, device=device) < self.channel_prob).float()
        
        if mask.sum() == 0:
            return x

        # 2. Calculate Min/Max per Channel (Vectorized)
        # Flatten spatial dims to find min/max easily
        x_flat = x.view(N, C, -1)
        
        # Shape: (N, C, 1)
        current_min = x_flat.min(dim=-1, keepdim=True)[0].view(N, C, 1)
        current_max = x_flat.max(dim=-1, keepdim=True)[0].view(N, C, 1)
        
        # Expand to (N, C, L) or (N, C, H, W)
        shape_diff = x.ndim - current_min.ndim
        for _ in range(shape_diff):
            current_min = current_min.unsqueeze(-1)
            current_max = current_max.unsqueeze(-1)
            mask = mask.unsqueeze(-1)

        # 3. Apply Scaling
        eps = 1e-8
        numerator = x - current_min
        denominator = (current_max - current_min) + eps
        
        # Normalized to 0..1
        x_norm = numerator / denominator
        
        # Scaled to target range
        scale = self.max_val - self.min_val
        x_scaled = x_norm * scale + self.min_val
        
        # 4. Blend based on mask
        # Where mask is 1, use scaled. Where mask is 0, use original.
        return mask * x_scaled + (1.0 - mask) * x


class RandomClamp(torch.nn.Module):
    """
    Randomly clamps (clips) values of selected channels.
    Simulates sensor saturation (e.g. signal hits the rail).
    """

    def __init__(
        self, 
        probability: float = 1.0, 
        min_val: float = -1.0, 
        max_val: float = 1.0, 
        channel_prob: float = 0.5
    ):
        """
        Parameters
        ----------
        probability : float
            Global probability.
        min_val, max_val : float
            Clamping limits. Values outside this range are clipped.
        channel_prob : float
            Probability per channel to be clamped.
        """
        super().__init__()
        self.probability = probability
        self.min_val = min_val
        self.max_val = max_val
        self.channel_prob = channel_prob

    def extra_repr(self) -> str:
        return f"prob={self.probability}, limit=[{self.min_val}, {self.max_val}], ch_prob={self.channel_prob}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input (N, C, L).
        """
        if torch.rand(1, device=x.device) > self.probability:
            return x
            
        N, C = x.shape[:2]
        device = x.device

        # 1. Determine active channels
        # Shape (N, C, 1...)
        mask_shape = [N, C] + [1] * (x.ndim - 2)
        mask = (torch.rand(*mask_shape, device=device) < self.channel_prob).float()
        
        if mask.sum() == 0:
            return x

        # 2. Compute Clamped Version
        # We clamp the *whole* tensor (cheap op) and then mix
        x_clamped = torch.clamp(x, self.min_val, self.max_val)
        
        # 3. Apply Mask
        # Only keep clamped values where mask is 1
        return mask * x_clamped + (1.0 - mask) * x


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # ---------------------------------------------------------
    # Test Data: 2 Channels
    # ---------------------------------------------------------
    t = torch.linspace(0, 10, 200)
    # Ch0: Sine with amplitude 5
    # Ch1: Sine with amplitude 5 (will be clamped)
    signal = (torch.sin(t) * 5).repeat(1, 2, 1) # (1, 2, 200)
    
    # 1. Test Scale
    # Force scaling Ch0 to [0, 1]
    # We cheat the probability to ensure we see the effect
    aug_scale = RandomChannelScaling(probability=1.0, min_val=0.0, max_val=1.0, channel_prob=1.0)
    out_scale = aug_scale(signal.clone())

    # 2. Test Clamp
    # Clamp between [-2, 2]. Original peaks are at 5/-5, so they should be cut off.
    aug_clamp = RandomClamp(probability=1.0, min_val=-2.0, max_val=2.0, channel_prob=1.0)
    out_clamp = aug_clamp(signal.clone())

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------

    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    
    # Original
    axs[0].plot(signal[0, 0].numpy(), label="Original (Amp=5)")
    axs[0].set_title("Original Signal")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    
    # Scaled
    axs[1].plot(out_scale[0, 0].numpy(), color='green', label="Scaled to [0, 1]")
    axs[1].set_title("RandomChannelScaling")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    
    # Clamped
    axs[2].plot(signal[0, 0].numpy(), 'k--', alpha=0.3, label="Original")
    axs[2].plot(out_clamp[0, 0].numpy(), color='red', label="Clamped to [-2, 2]")
    axs[2].set_title("RandomClamp (Saturation Simulation)")
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    print("Scaling and Clamping tests done.")