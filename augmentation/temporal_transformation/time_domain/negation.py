# -*- coding: utf-8 -*-
"""
Random Negation Augmentation.

Inverts the sign of the signal (multiplication by -1).
Physically corresponds to flipping the signal polarity or electrical phase.

@author: Diyar Altinses, M.Sc.
"""

import torch


class RandomNegation(torch.nn.Module):
    """
    Randomly inverts the sign of the input tensor.
    Formula: x = -x
    
    This preserves the magnitude (energy) but flips the phase by 180 degrees.
    """

    def __init__(self, probability: float = 0.5):
        """
        Parameters
        ----------
        probability : float
            Probability of applying the negation per sample.
            0.5 is usually ideal to teach the model sign-invariance.
        """
        super().__init__()
        self.probability = probability

    def extra_repr(self) -> str:
        return f"probability={self.probability}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor. Shape (N, C, L) or (N, C, H, W).
            
        Returns:
            torch.Tensor: Randomly negated tensor.
        """
        # 1. Global Probability Check (Optimization)
        # If prob is 0, do nothing. If 1, negate everything.
        if self.probability == 0.0:
            return x
        if self.probability == 1.0:
            return -x

        # 2. Vectorized Per-Sample Decision
        N = x.shape[0]
        device = x.device
        
        # Generate random mask: True where we want to negate
        # Shape: (N,)
        do_negate = torch.rand(N, device=device) < self.probability
        
        # If no samples selected, return early
        if not do_negate.any():
            return x

        # 3. Create Sign Tensor for Broadcasting
        # We need a tensor of shape (N, 1, 1...) filled with 1.0 or -1.0
        # 1.0 = Keep, -1.0 = Negate
        signs = torch.where(do_negate, -1.0, 1.0)
        
        # Reshape for broadcasting
        # We append 1s for all dimensions after Batch
        view_shape = [N] + [1] * (x.ndim - 1)
        signs = signs.view(*view_shape)
        
        # 4. Apply
        return x * signs


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # ---------------------------------------------------------
    # Test Data: Sine Wave
    # Shape: (Batch=3, Channels=1, Length=100)
    # ---------------------------------------------------------
    t = torch.linspace(0, 6.28, 100)
    # Create 3 identical sine waves
    batch = torch.sin(t).view(1, 1, 100).repeat(3, 1, 1)
    
    # Augment: 50% chance to flip each signal
    aug = RandomNegation(probability=0.5)
    out = aug(batch)

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------
    

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    colors = ['blue', 'orange', 'green']
    for i in range(3):
        # Check if flipped
        is_flipped = (out[i, 0, 10] < 0 and batch[i, 0, 10] > 0)
        status = "FLIPPED (-1)" if is_flipped else "Original (+1)"
        
        ax.plot(t.numpy(), out[i, 0].numpy(), 
                color=colors[i], label=f"Sample {i}: {status}", linewidth=2, alpha=0.7)
        
    ax.set_title("Random Negation (Per-Sample)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    print("RandomNegation test done.")