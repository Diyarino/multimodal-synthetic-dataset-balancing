# -*- coding: utf-8 -*-
"""
Nebula / Fog Augmentation.

Simulates atmospheric scattering (fog/haze) using Koschmieder's Law.
Uses a vertical gradient to simulate depth (ground plane assumption).

@author: Diyar Altinses, M.Sc.
"""

import torch


class Nebula(torch.nn.Module):
    """
    Applies synthesized fog to an image based on a synthetic depth gradient.
    
    Model: Koschmieder's Law
    I(x) = J(x)t(x) + A(1-t(x))
    where:
        I = Observed Image
        J = Clear Image
        A = Atmospheric Light (Luminance)
        t = Transmission = exp(-beta * distance)
    """

    def __init__(
        self, 
        probability: float = 0.5, 
        visibility: float = 512.0, 
        max_distance: float = 256.0, 
        atmosphere: float = 255.0
    ):
        """
        Parameters
        ----------
        probability : float
            Probability of applying the fog effect.
        visibility : float
            Meteorological visibility range (V). 
            Higher = less fog. The extinction coefficient is derived as approx 3.0 / V.
        max_distance : float
            Simulated distance at the horizon (top of image).
            Controls the steepness of the fog gradient.
        atmosphere : float
            Intensity of the "sky" or fog color (A). 
            Default 255 assumes byte images. For 0-1 float images, this is auto-scaled.
        """
        super().__init__()
        self.probability = probability
        self.visibility = visibility
        self.max_distance = max_distance
        self.atmosphere = atmosphere

    def extra_repr(self) -> str:
        return f"prob={self.probability}, visibility={self.visibility}, max_dist={self.max_distance}, atm={self.atmosphere}"

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): Input image (C, H, W) or (N, C, H, W).
            
        Returns:
            torch.Tensor: Foggy image.
        """
        # 1. Probability Check
        if torch.rand(1, device=img.device) > self.probability:
            return img

        # 2. Setup Dimensions
        is_batch = img.ndim == 4
        x = img if is_batch else img.unsqueeze(0)
        N, C, H, W = x.shape
        device = x.device

        # 3. Handle Range (0-1 vs 0-255)
        # If the image is 0-1 but user passed atmosphere=255, we must adapt A.
        # Simple heuristic: if input max <= 1.0, scale atmosphere down.
        is_float_range = x.max() <= 1.0
        A = self.atmosphere
        if is_float_range and A > 1.0:
            A = 1.0  # Cap atmosphere at white

        # 4. Generate Synthetic Depth Map (Vertical Gradient)
        # We assume the top of the image is "far" (distance = max) and bottom is "close" (0).
        # Shape: (H,)
        # Creates linear gradient from max_dist down to 0
        depth_profile = torch.linspace(self.max_distance, 0, steps=H, device=device)
        
        # Broadcast to (N, C, H, W)
        # We need shape (1, 1, H, 1) to broadcast correctly against W
        depth_map = depth_profile.view(1, 1, H, 1)

        # 5. Calculate Transmission (t)
        # Koschmieder's coefficient beta ~ 3.0 / visibility (assuming 5% contrast threshold)
        # Formula: t = exp(-beta * d)
        beta = 2.996 / (self.visibility + 1e-6) # ln(20) approx 2.996
        
        # transmission map will broadcast depth_map along Width
        transmission = torch.exp(-beta * depth_map)

        # 6. Apply Fog Formula
        # Out = Input * t + Atmosphere * (1 - t)
        # x is (N, C, H, W), transmission is (1, 1, H, 1) -> broadcasts automatically
        out = x * transmission + A * (1.0 - transmission)

        # 7. Restore Types
        if img.dtype == torch.uint8:
            out = out.clamp(0, 255).to(torch.uint8)
        else:
            out = out.clamp(0, A if A <= 1.0 else 255.0)

        return out if is_batch else out.squeeze(0)


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # 1. Test with a Float Image (0-1) - typically what Neural Networks use
    # Create a dummy landscape: Blue sky top, green ground bottom
    H, W = 256, 256
    img_float = torch.zeros(3, H, W)
    
    # Simple Gradient Background
    y_grid = torch.linspace(0, 1, H).view(1, H, 1).expand(3, H, W)
    img_float[0] = 0.5 - y_grid[0]*0.5 # Blueish top
    img_float[1] = y_grid[1] # Greenish bottom
    img_float[2] = 0.2
    
    # 2. Apply Fog
    # Visibility 200 means very dense fog.
    # Atmosphere 1.0 means white fog.
    aug = Nebula(probability=1.0, visibility=150.0, max_distance=300.0, atmosphere=1.0)
    
    out = aug(img_float)

    # Visualization
    

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    
    axs[0].imshow(img_float.permute(1, 2, 0).clip(0, 1))
    axs[0].set_title("Original (Synthetic Landscape)")
    axs[0].axis('off')
    
    axs[1].imshow(out.permute(1, 2, 0).clip(0, 1))
    axs[1].set_title("Augmented (Foggy)")
    axs[1].axis('off')
    
    plt.tight_layout()
    plt.show()