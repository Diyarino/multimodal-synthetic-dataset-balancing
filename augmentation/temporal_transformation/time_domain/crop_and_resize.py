# -*- coding: utf-8 -*-
"""
Crop and Resize (Zoom) Augmentation.

Simulates "Zooming In" on a time series or image.
1. Upsamples the signal (making it longer).
2. Randomly crops a segment of the original length.
Effect: The signal appears 'slower' or 'stretched' in time.

@author: Diyar Altinses, M.Sc.
"""

import torch


class CropAndResize(torch.nn.Module):
    """
    Randomly crops a portion of the signal and resizes it back to the original length.
    Or, equivalently: Upsamples the signal and crops to original length.
    
    Effect: "Zoom in" on a time segment.
    """

    def __init__(self, probability: float = 1.0, zoom_factor: float = 2.0):
        """
        Parameters
        ----------
        probability : float
            Probability of applying the augmentation.
        zoom_factor : float
            How much to zoom in.
            2.0 = Signal becomes 2x slower (interpolated to double length, then cropped).
            1.5 = Signal becomes 1.5x slower.
        """
        super().__init__()
        self.probability = probability
        self.zoom_factor = zoom_factor

    def extra_repr(self) -> str:
        return f"probability={self.probability}, zoom_factor={self.zoom_factor}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor. Shape (N, C, L).
            
        Returns:
            torch.Tensor: Augmented tensor of same shape (N, C, L).
        """
        # 1. Probability Check
        if torch.rand(1, device=x.device) > self.probability:
            return x

        # 2. Dimensions
        # Expect (N, C, L)
        if x.ndim == 2:
            x = x.unsqueeze(1) # (N, L) -> (N, 1, L)
        
        N, C, L = x.shape
        device = x.device

        # 3. Upsample (Resize)
        # We stretch the signal to new_length = L * zoom
        new_length = int(L * self.zoom_factor)
        
        # F.interpolate expects (N, C, L)
        # Use 'linear' for 1D signals
        upsampled = torch.nn.functional.interpolate(
            x, 
            size=new_length, 
            mode='linear', 
            align_corners=False
        )

        # 4. Random Crop back to original length L
        # We need to pick a start index between [0, new_length - L]
        max_start = new_length - L
        
        # If we want independent crops for each sample in batch (Vectorized crop is tricky without loop or advanced indexing)
        # For simplicity and speed in standard data loaders, a unified crop or simple loop is used.
        # Let's do a loop-free version using gathering if N is large, 
        # but for typical batch sizes, generating one start index per sample is best.
        
        starts = torch.randint(0, max_start + 1, (N,), device=device)
        
        # Advanced Indexing to crop different windows per sample:
        # Create grid: (N, L)
        grid = torch.arange(L, device=device).unsqueeze(0).expand(N, L)
        # Shift grid by starts: (N, L)
        indices = grid + starts.unsqueeze(1)
        
        # Expand for channels: (N, C, L)
        indices = indices.unsqueeze(1).expand(N, C, L)
        
        # Gather
        # upsampled is (N, C, new_length)
        # indices is (N, C, L)
        out = torch.gather(upsampled, 2, indices)

        return out


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # ---------------------------------------------------------
    # Test: Sine Wave
    # ---------------------------------------------------------
    t = torch.linspace(0, 4 * 3.14159, 100) # 2 cycles of sine
    signal = torch.sin(t).view(1, 1, 100)
    
    # Augment: Zoom 2x (should show only ~1 cycle instead of 2, stretched)
    aug = CropAndResize(probability=1.0, zoom_factor=2.0)
    out = aug(signal.clone())

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    
    ax.plot(t.numpy(), signal[0, 0].numpy(), 'k--', alpha=0.5, label="Original (2 Cycles)")
    # We plot output against same 't' to show it fills the same time window
    ax.plot(t.numpy(), out[0, 0].numpy(), 'b-', label="Crop & Resize (Zoomed In)")
    
    ax.set_title("CropAndResize Augmentation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    print("CropAndResize test done.")