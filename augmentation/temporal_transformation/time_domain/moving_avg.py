# -*- coding: utf-8 -*-
"""
Smoothing & Moving Average Augmentations.

Reduces noise and high-frequency details.
1. MovingAverage: Standard Box Filter (Uniform weights).
2. RandomSmoothing: Gaussian-like smoothing with dynamic kernel weights.

@author: Diyar Altinses, M.Sc.
"""

import torch


class MovingAverage(torch.nn.Module):
    """
    Applies a Moving Average (Box Blur) filter.
    Optimized using Average Pooling.
    """

    def __init__(self, probability: float = 1.0, window_size: int = 3):
        """
        Parameters
        ----------
        probability : float
            Probability of applying the augmentation.
        window_size : int
            Size of the smoothing window (kernel size). 
            Should be odd (3, 5, 7...) for symmetry.
        """
        super().__init__()
        self.probability = probability
        self.window_size = window_size
        # Calculate padding to keep output size == input size
        self.padding = window_size // 2

    def extra_repr(self) -> str:
        return f"probability={self.probability}, window_size={self.window_size}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor (N, C, L).
        """
        if torch.rand(1, device=x.device) > self.probability:
            return x

        # Ensure 3D input (N, C, L)
        if x.ndim == 2:
            x_in = x.unsqueeze(1)
        else:
            x_in = x

        # Apply Average Pooling
        # stride=1 makes it a sliding window
        # padding ensures 'same' output length
        out = torch.nn.functional.avg_pool1d(
            x_in, 
            kernel_size=self.window_size, 
            stride=1, 
            padding=self.padding, 
            count_include_pad=False
        )

        # Handle even window sizes (rare case where size changes by 1)
        if out.shape[-1] != x_in.shape[-1]:
            out = torch.nn.functional.interpolate(out, size=x_in.shape[-1], mode='linear', align_corners=False)

        # Restore original shape if we unsqueezed
        if x.ndim == 2:
            out = out.squeeze(1)

        return out


class RandomSmoothing(torch.nn.Module):
    """
    Applies a random 3-tap smoothing filter.
    Kernel weights are dynamic: [factor/2, 1-factor, factor/2].
    """

    def __init__(self, probability: float = 1.0, factor_range: tuple = (0.0, 0.5)):
        super().__init__()
        self.probability = probability
        self.factor_min = factor_range[0]
        self.factor_max = factor_range[1]

    def extra_repr(self) -> str:
        return f"probability={self.probability}, factor_range=({self.factor_min}, {self.factor_max})"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1, device=x.device) > self.probability:
            return x

        # Dimensions handling
        if x.ndim == 2:
            x_in = x.unsqueeze(1)
            was_2d = True
        else:
            x_in = x
            was_2d = False
            
        N, C, L = x_in.shape
        device = x.device

        # 1. Generate Random Factor
        factor = torch.empty(1, device=device).uniform_(self.factor_min, self.factor_max).item()

        # 2. Create Kernel
        w_side = factor / 2
        w_center = 1.0 - factor
        
        # Shape (C, 1, 3) -> Output Channels, Input Channels/Groups, Length
        kernel = torch.tensor([[[w_side, w_center, w_side]]], device=device)
        kernel = kernel.repeat(C, 1, 1)

        # 3. Apply Padding Manually (Robust way)
        # Pad 1 pixel on left and 1 on right with 'replicate' mode
        # (replicate repeats the edge value, preventing black borders)
        x_padded = torch.nn.functional.pad(x_in, (1, 1), mode='replicate')

        # 4. Apply Convolution
        # We set padding=0 inside conv1d because we already padded manually.
        # We explicitly pass bias=None to keep PyTorch happy.
        out = torch.nn.functional.conv1d(
            x_padded, 
            weight=kernel, 
            bias=None, 
            stride=1, 
            padding=0, 
            groups=C
        )

        if was_2d:
            out = out.squeeze(1)
            
        return out

# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # ---------------------------------------------------------
    # Test Data: Noisy Signal
    # ---------------------------------------------------------
    t = torch.linspace(0, 10, 100)
    # Sine wave + Gaussian Noise
    signal = torch.sin(t) + torch.randn(100) * 0.2
    signal = signal.view(1, 1, 100)
    
    # 1. Moving Average (Box Filter)
    # Window 5 -> Strong smoothing
    aug_avg = MovingAverage(probability=1.0, window_size=5)
    out_avg = aug_avg(signal.clone())

    # 2. Random Smoothing (Gaussian-like)
    # Factor 0.8 -> Strong smoothing
    aug_smooth = RandomSmoothing(probability=1.0, factor_range=(0.7, 0.9))
    out_smooth = aug_smooth(signal.clone())

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Moving Average
    axs[0].plot(t.numpy(), signal[0, 0].numpy(), 'k', alpha=0.3, label="Noisy Input")
    axs[0].plot(t.numpy(), out_avg[0, 0].numpy(), 'b', label="Moving Average (Win=5)")
    axs[0].set_title("Moving Average (Box Filter)")
    axs[0].legend()
    
    # Random Smoothing
    axs[1].plot(t.numpy(), signal[0, 0].numpy(), 'k', alpha=0.3, label="Noisy Input")
    axs[1].plot(t.numpy(), out_smooth[0, 0].numpy(), 'r', label="Random Smoothing (Kernel Conv)")
    axs[1].set_title("Random Smoothing (Soft Blur)")
    axs[1].legend()
    
    plt.tight_layout()
    plt.show()
    print("Smoothing test done.")