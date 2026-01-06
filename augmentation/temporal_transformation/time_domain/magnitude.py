# -*- coding: utf-8 -*-
"""
Random Rectification Augmentation.

Applies absolute value function (Full-Wave Rectification) to the signal.
Optionally inverts the result to be fully negative.
Useful for studying envelope characteristics regardless of sign.

@author: Diyar Altinses, M.Sc.
"""

import torch


class RandomRectify(torch.nn.Module):
    """
    Applies full-wave rectification to the input (Absolute Value).
    With a certain probability, flips the result to be negative.
    
    Result is always fully positive OR fully negative. 
    Original sign information is lost.
    """

    def __init__(self, probability: float = 1.0, negative_prob: float = 0.5):
        """
        Parameters
        ----------
        probability : float
            Probability of applying the augmentation (taking the absolute value).
        negative_prob : float
            Given that the augmentation is applied, probability that the result 
            will be inverted (negative absolute value).
            0.0 = Always Positive (Standard abs).
            1.0 = Always Negative (-abs).
            0.5 = Randomly Positive or Negative.
        """
        super().__init__()
        self.probability = probability
        self.negative_prob = negative_prob

    def extra_repr(self) -> str:
        return f"probability={self.probability}, negative_prob={self.negative_prob}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor. Shape (N, C, L) or (N, C, H, W).
            
        Returns:
            torch.Tensor: Rectified tensor.
        """
        # 1. Probability Check
        if torch.rand(1, device=x.device) > self.probability:
            return x

        # 2. Apply Absolute Value (Rectification)
        # Result is now all positive [0, +inf]
        x_rect = x.abs()
        
        # 3. Determine Signs per Sample (Vectorized)
        # We want to decide for each sample in the batch whether to flip it or not.
        N = x.shape[0]
        
        # Generate random floats for each sample
        # Shape: (N, 1, 1...) for broadcasting
        # We need to reshape mask to match input dimensions (except batch)
        view_shape = [N] + [1] * (x.ndim - 1)
        
        random_vals = torch.rand(N, device=x.device).view(*view_shape)
        
        # Create Sign Tensor:
        # If random < negative_prob -> -1.0
        # Else -> +1.0
        signs = torch.where(random_vals < self.negative_prob, -1.0, 1.0)
        
        # 4. Apply Signs
        return x_rect * signs


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # ---------------------------------------------------------
    # Test Data: Sine Wave (goes positive and negative)
    # Shape: (Batch=3, Channels=1, Length=100)
    # ---------------------------------------------------------
    t = torch.linspace(0, 4 * 3.14, 100)
    signal = torch.sin(t).view(1, 1, 100).repeat(3, 1, 1)
    
    # Augment: 
    # Probability 1.0 -> Always rectify.
    # Negative Prob 0.5 -> roughly half the batch should be negative.
    aug = RandomRectify(probability=1.0, negative_prob=0.5)
    out = aug(signal.clone())

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------
    fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True, sharey=True)
    
    for i in range(3):
        axs[i].plot(t.numpy(), signal[i, 0].numpy(), 'k--', alpha=0.3, label="Original")
        axs[i].plot(t.numpy(), out[i, 0].numpy(), 'b-', label="Rectified")
        
        # Determine if it became positive or negative
        is_neg = out[i, 0].mean() < 0
        status = "Negative (-|x|)" if is_neg else "Positive (|x|)"
        
        axs[i].set_title(f"Sample {i}: {status}")
        axs[i].legend(loc='upper right')
        axs[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    print("RandomRectify test done.")