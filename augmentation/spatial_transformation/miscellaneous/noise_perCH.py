# -*- coding: utf-8 -*-
"""
Signal-Dependent Noise Augmentations.

Scales the noise amplitude based on the standard deviation of the input signal
along a specific dimension. Useful for sensors where noise correlates with signal complexity.

@author: Diyar Altinses, M.Sc.
"""

import torch


class GaussianNoisePerChannel(torch.nn.Module):
    """
    Adds Gaussian noise scaled by the standard deviation of the input along a dimension.
    Formula: Out = In + N(mean, std) * Input_Std(dim) * p_std
    """

    def __init__(
        self, 
        probability: float = 1.0, 
        mean: float = 0.0, 
        p_std: float = 1.0, 
        dim: int = -1
    ):
        """
        Parameters
        ----------
        probability : float
            Probability of applying the noise.
        mean : float
            Base mean of the added noise (offset).
        p_std : float
            Scaling factor for the noise. 
            Final Noise Std = Input_Std_along_dim * p_std.
        dim : int
            The dimension along which to calculate the input's standard deviation.
            Default -1 (Width).
        """
        super().__init__()
        self.probability = probability
        self.mean = mean
        self.p_std = p_std
        self.dim = dim

    def extra_repr(self) -> str:
        return f"probability={self.probability}, mean={self.mean}, p_std={self.p_std}, dim={self.dim}"

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(1, device=img.device) > self.probability:
            return img

        # 1. Calculate Signal Standard Deviation
        # Shape: If img is (N, C, H, W) and dim is -1, std is (N, C, H)
        inp_std = img.float().std(dim=self.dim, keepdim=True)
        
        # 2. Generate Noise
        # noise ~ N(0, 1) * signal_std * scale + mean
        noise = torch.randn_like(img) * inp_std * self.p_std + self.mean
        
        # 3. Add
        out = img + noise

        # 4. Restore Type (if needed)
        if not img.is_floating_point():
            out = out.to(img.dtype)
            
        return out


class WhiteNoisePerChannel(torch.nn.Module):
    """
    Adds Uniform (White) noise scaled by the standard deviation of the input.
    Formula: Out = In + Uniform(-0.5, 0.5) * Input_Std(dim) * amplitude
    """

    def __init__(
        self, 
        probability: float = 1.0, 
        amplitude: float = 1.0, 
        dim: int = -1
    ):
        """
        Parameters
        ----------
        probability : float
            Probability of applying the noise.
        amplitude : float
            Scaling factor for the noise range.
        dim : int
            Dimension to calculate signal standard deviation.
        """
        super().__init__()
        self.probability = probability
        self.amplitude = amplitude
        self.dim = dim

    def extra_repr(self) -> str:
        return f"probability={self.probability}, amplitude={self.amplitude}, dim={self.dim}"

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(1, device=img.device) > self.probability:
            return img

        # 1. Calculate Signal Standard Deviation
        inp_std = img.float().std(dim=self.dim, keepdim=True)
        
        # 2. Generate White Noise (Centered)
        # torch.rand is [0, 1]. We want [-0.5, 0.5] to avoid brightness shift.
        noise_base = (torch.rand_like(img) - 0.5)
        
        # Scale by signal std
        noise = noise_base * inp_std * self.amplitude
        
        # 3. Add
        out = img + noise

        if not img.is_floating_point():
            out = out.to(img.dtype)

        return out


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # Create a synthetic image with varying variance
    # Top half: Flat gray (Low Variance -> Low Noise)
    # Bottom half: High frequency stripes (High Variance -> High Noise)
    H, W = 100, 100
    img = torch.zeros(1, 1, H, W)
    img[0, 0, :50, :] = 0.5
    img[0, 0, 50:, ::2] = 0.8
    img[0, 0, 50:, 1::2] = 0.2
    
    # Augment
    # We use dim=-1 (Width). 
    # Top half has std=0 along width. Bottom half has high std.
    aug_gauss = GaussianNoisePerChannel(probability=1.0, p_std=2.0, dim=-1)
    res = aug_gauss(img)
    
    

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    
    axs[0].imshow(img[0, 0], cmap='gray', vmin=0, vmax=1)
    axs[0].set_title("Original\n(Top: Flat, Bottom: Textured)")
    axs[0].axis('off')
    
    axs[1].imshow(res[0, 0], cmap='gray', vmin=0, vmax=1)
    axs[1].set_title("Result\n(Noise appears only on texture!)")
    axs[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    print("Per-Channel Noise test done.")