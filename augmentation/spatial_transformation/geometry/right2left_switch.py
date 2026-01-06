# -*- coding: utf-8 -*-
"""
Random Roll / Cyclic Shift Augmentation.

Shifts the elements of the tensor cyclically along a dimension.
Elements that roll off one end reappear at the other end.

@author: Diyar Altinses, M.Sc.
"""

import torch


class RandomRoll(torch.nn.Module):
    """
    Performs a cyclic shift (roll) on the input tensor.
    Useful for creating invariance to translation where boundaries wrap around
    (e.g. 360-degree panorama images or periodic signals).
    """

    def __init__(self, probability: float = 1.0, fraction: float = 0.2, dim: int = -1):
        """
        Parameters
        ----------
        probability : float
            Probability of applying the roll.
        fraction : float
            Fraction of the dimension size to shift (0.0 to 1.0).
            Example: 0.2 on a width of 100 pixels means a shift of 20 pixels.
        dim : int
            The dimension to roll along.
            -1 = Width (Left/Right)
            -2 = Height (Up/Down)
        """
        super().__init__()
        self.probability = probability
        self.fraction = fraction
        self.dim = dim

    def extra_repr(self) -> str:
        return f"probability={self.probability}, fraction={self.fraction}, dim={self.dim}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor. Shape (..., H, W).
            
        Returns:
            torch.Tensor: Rolled tensor.
        """
        # 1. Probability Check
        if torch.rand(1, device=x.device) > self.probability:
            return x

        # 2. Calculate Shift Amount
        # Get size of the target dimension
        size = x.shape[self.dim]
        shift = int(size * self.fraction)
        
        # 3. Determine Direction (Left or Right)
        # Randomly choose +shift or -shift
        if torch.rand(1, device=x.device) < 0.5:
            shift = -shift

        # 4. Apply Roll
        # torch.roll is cleaner and often faster than slicing/cat
        return torch.roll(x, shifts=shift, dims=self.dim)


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # Create a dummy image with a distinct pattern (Gradient)
    # Shape: (1, 1, 100, 100) -> Batch, Channel, Height, Width
    H, W = 100, 100
    y, x = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing='ij')
    img = x.unsqueeze(0).unsqueeze(0) # Gradient increases from left to right
    
    # 1. Roll Horizontal (Left/Right)
    # fraction 0.3 means 30% shift
    aug = RandomRoll(probability=1.0, fraction=0.3, dim=-1)
    res = aug(img)

    # Visualization
    #
    # Use standard matplotlib to visualize the roll effect
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    
    axs[0].imshow(img[0, 0], cmap='viridis')
    axs[0].set_title("Original (Gradient 0->1)")
    axs[0].axis('off')
    
    axs[1].imshow(res[0, 0], cmap='viridis')
    axs[1].set_title("Rolled (Wrap Around)")
    axs[1].axis('off')
    
    # Draw a line to show the wrap point
    axs[1].axvline(x=30, color='white', linestyle='--', alpha=0.5)
    axs[1].axvline(x=70, color='white', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    print("RandomRoll test done.")