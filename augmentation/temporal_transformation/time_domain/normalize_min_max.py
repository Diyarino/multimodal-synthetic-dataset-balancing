# -*- coding: utf-8 -*-
"""
Min-Max Normalization Augmentation.

Rescales the data to a specific range (e.g., [0, 1] or [-1, 1]).
Essential for preprocessing neural network inputs.

@author: Diyar Altinses, M.Sc.
"""

import torch


class NormalizeMinMax(torch.nn.Module):
    """
    Normalizes the input tensor to a target range [min, max].
    
    Formula: X_norm = (X - X.min) / (X.max - X.min) * (target_max - target_min) + target_min
    """

    def __init__(
        self, 
        probability: float = 1.0, 
        min_val: float = 0.0, 
        max_val: float = 1.0,
        per_sample: bool = True,
        per_channel: bool = False
    ):
        """
        Parameters
        ----------
        probability : float
            Probability of applying the normalization.
        min_val, max_val : float
            Target range. Default [0, 1].
        per_sample : bool
            If True, calculates min/max for each sample in the batch independently.
            If False, calculates global min/max across the whole batch.
        per_channel : bool
            If True, calculates min/max for each channel independently (e.g. RGB distinct).
            If False, scales all channels by the sample's global min/max.
        """
        super().__init__()
        self.probability = probability
        self.min_val = min_val
        self.max_val = max_val
        self.per_sample = per_sample
        self.per_channel = per_channel

    def extra_repr(self) -> str:
        return f"prob={self.probability}, range=[{self.min_val}, {self.max_val}], per_sample={self.per_sample}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor. Shape (N, C, L) or (N, C, H, W).
        """
        # 1. Probability Check
        if torch.rand(1, device=x.device) > self.probability:
            return x

        # 2. Determine reduction dimensions
        # We want to find min/max along these dimensions.
        # x shape: (N, C, ...)
        
        dims_to_reduce = []
        
        # Always reduce spatial/time dimensions (last dims)
        if x.ndim >= 2:
            dims_to_reduce.extend(list(range(2, x.ndim)))
        
        # If NOT per_channel, we also reduce across channel dimension (dim 1)
        if not self.per_channel and x.ndim >= 2:
            dims_to_reduce.append(1)
            
        # If NOT per_sample, we also reduce across batch dimension (dim 0)
        if not self.per_sample:
            dims_to_reduce.append(0)
            
        # If dims_to_reduce is empty (e.g. scalar input), handle gracefully
        if not dims_to_reduce:
            return torch.clamp(x, self.min_val, self.max_val) # Fallback

        # 3. Calculate Min and Max
        # Note: torch.min/max over multiple dims is tricky in older pytorch.
        # Efficient way: Flatten the reduction dims first.
        
        # Identify dims to keep
        keep_dims = []
        if self.per_sample: keep_dims.append(0)
        if self.per_channel: keep_dims.append(1)
        
        # We use a view trick to calculate min/max
        # Reshape to (Keep_Dims..., -1)
        view_shape = [x.shape[i] for i in keep_dims] + [-1]
        x_flat = x.view(*view_shape)
        
        # Min/Max along the last flattened dimension
        current_min = x_flat.min(dim=-1, keepdim=True)[0]
        current_max = x_flat.max(dim=-1, keepdim=True)[0]
        
        # 4. Reshape for Broadcasting
        # We need to expand current_min back to x.shape
        # Example: x is (N, C, L), min is (N, 1). We need (N, 1, 1).
        
        # Construct the shape for broadcasting
        # Start with the shape of min/max (e.g. N, 1)
        broadcast_shape = list(current_min.shape[:-1]) 
        
        # Add 1s for reduced dimensions
        # If x is (N, C, L), and we kept N, C -> broadcast (N, C, 1)
        # If x is (N, C, L), and we kept N -> broadcast (N, 1, 1)
        target_ndim = x.ndim
        while len(broadcast_shape) < target_ndim:
            broadcast_shape.append(1)
            
        current_min = current_min.view(broadcast_shape)
        current_max = current_max.view(broadcast_shape)

        # 5. Normalize
        # Add epsilon to avoid division by zero
        eps = 1e-8
        numerator = x - current_min
        denominator = (current_max - current_min) + eps
        
        x_norm = numerator / denominator
        
        # 6. Scale to target range
        scale = self.max_val - self.min_val
        out = x_norm * scale + self.min_val
        
        return out


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # ---------------------------------------------------------
    # Test Data: 2 Samples with very different ranges
    # ---------------------------------------------------------
    # Sample 0: Range [0, 10]
    # Sample 1: Range [1000, 2000]
    L = 100
    batch = torch.zeros(2, 1, L)
    batch[0, 0] = torch.linspace(0, 10, L)
    batch[1, 0] = torch.linspace(1000, 2000, L)
    
    # Augment: Normalize EACH sample individually to [0, 1]
    aug = NormalizeMinMax(probability=1.0, min_val=0.0, max_val=1.0, per_sample=True)
    out = aug(batch)

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    
    # Plot Original
    axs[0, 0].plot(batch[0, 0].numpy())
    axs[0, 0].set_title("Original Sample 0 (Range 0-10)")
    
    axs[0, 1].plot(batch[1, 0].numpy())
    axs[0, 1].set_title("Original Sample 1 (Range 1000-2000)")
    
    # Plot Normalized
    axs[1, 0].plot(out[0, 0].numpy())
    axs[1, 0].set_title("Norm Sample 0 (Range 0-1)")
    axs[1, 0].set_ylim(-0.1, 1.1)
    
    axs[1, 1].plot(out[1, 0].numpy())
    axs[1, 1].set_title("Norm Sample 1 (Range 0-1)")
    axs[1, 1].set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.show()
    print("NormalizeMinMax test done.")