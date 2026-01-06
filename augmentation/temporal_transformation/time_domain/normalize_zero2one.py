# -*- coding: utf-8 -*-
"""
Zero-One Normalization Augmentation.

Special case of Min-Max normalization where the target range is strictly [0, 1].
This is the standard scaling method for images (0-255 -> 0-1).

@author: Diyar Altinses, M.Sc.
"""

import torch

class NormalizeZeroOne(torch.nn.Module):
    """
    Scales the input tensor to the range [0, 1].
    Formula: (x - min) / (max - min + eps)
    """

    def __init__(
        self, 
        probability: float = 1.0, 
        per_sample: bool = True,
        per_channel: bool = False,
        eps: float = 1e-8
    ):
        """
        Parameters
        ----------
        probability : float
            Probability of applying the normalization.
        per_sample : bool
            If True, calculates min/max for each sample independently (Standard).
            If False, calculates global min/max across the batch.
        per_channel : bool
            If True, normalizes each channel independently (e.g. R, G, B separately).
        eps : float
            Small value to avoid division by zero if max == min.
        """
        super().__init__()
        self.probability = probability
        self.per_sample = per_sample
        self.per_channel = per_channel
        self.eps = eps

    def extra_repr(self) -> str:
        return f"prob={self.probability}, per_sample={self.per_sample}, eps={self.eps}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor. Shape (N, C, L) or (N, C, H, W).
        """
        # 1. Probability Check
        if torch.rand(1, device=x.device) > self.probability:
            return x

        # 2. Determine Dimensions to Reduce
        # We want to find min/max along dimensions that are NOT N or C (depending on flags)
        
        keep_dims = []
        if self.per_sample: keep_dims.append(0)
        if self.per_channel: keep_dims.append(1)
        
        # If input is a scalar or empty, return
        if x.numel() == 0:
            return x

        # 3. Calculate Min/Max (Vectorized)
        # Reshape to (Keep_Dims..., -1) to reduce all other dims at once
        view_shape = [x.shape[i] for i in keep_dims] + [-1]
        x_flat = x.view(*view_shape)
        
        # Calculate min/max along the flattened dimension
        current_min = x_flat.min(dim=-1, keepdim=True)[0]
        current_max = x_flat.max(dim=-1, keepdim=True)[0]
        
        # 4. Restore Shapes for Broadcasting
        # Example: If x is (N, C, H, W) and we kept (N, C), we have (N, C, 1).
        # We need to expand to (N, C, 1, 1).
        
        broadcast_shape = list(current_min.shape[:-1]) 
        target_ndim = x.ndim
        
        while len(broadcast_shape) < target_ndim:
            broadcast_shape.append(1)
            
        current_min = current_min.view(broadcast_shape)
        current_max = current_max.view(broadcast_shape)

        # 5. Normalize
        # (x - min) / (max - min)
        numerator = x - current_min
        denominator = (current_max - current_min) + self.eps
        
        return numerator / denominator


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # ---------------------------------------------------------
    # Test Data: Signal with arbitrary offset and scale
    # ---------------------------------------------------------
    t = torch.linspace(0, 10, 100)
    # Range is roughly [-100, 100] + 50 = [-50, 150]
    signal = (torch.sin(t) * 100 + 50).view(1, 1, 100)
    
    aug = NormalizeZeroOne(probability=1.0)
    out = aug(signal)

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    
    # Original
    axs[0].plot(signal[0, 0].numpy())
    axs[0].set_title(f"Original\nMin: {signal.min():.1f}, Max: {signal.max():.1f}")
    axs[0].grid(True, alpha=0.3)
    
    # Normalized
    axs[1].plot(out[0, 0].numpy(), color='green')
    axs[1].set_title(f"Normalized [0, 1]\nMin: {out.min():.1f}, Max: {out.max():.1f}")
    axs[1].set_ylim(-0.1, 1.1)
    axs[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    print("NormalizeZeroOne test done.")