# -*- coding: utf-8 -*-
"""
Smoke Augmentation.

Overlays smoke/fog textures with alpha transparency onto the image.
Optimized for PyTorch pipelines using Alpha Compositing.

@author: Diyar Altinses, M.Sc.
"""

import logging
from pathlib import Path
from typing import Optional, List, Union
import torch
from torchvision.transforms.functional import resize, to_tensor
from PIL import Image

# Logging Setup
logger = logging.getLogger(__name__)


class Smoke(torch.nn.Module):
    """
    Overlays smoke images (PNG with Alpha channel) onto the input.
    Uses standard Alpha Compositing (Porter-Duff 'Source Over').
    """

    def __init__(
        self, 
        probability: float = 0.5, 
        path: Optional[Union[str, Path]] = None, 
        opacity: float = 1.0
    ):
        """
        Parameters
        ----------
        probability : float
            Probability of applying the smoke effect.
        path : str or Path, optional
            Path to the directory containing smoke images (PNGs).
            Default tries to locate '../resources/smoke'.
        opacity : float
            Global transparency factor for the smoke (0.0 to 1.0).
            Multiplies with the image's own alpha channel.
        """
        super().__init__()
        self.probability = probability
        self.opacity = opacity
        
        # Determine Path
        if path is None:
            # Robust relative path resolution
            self.overlay_path = Path(__file__).resolve().parent.parent / 'resources' / 'smoke'
        else:
            self.overlay_path = Path(path)

        # Load images into memory
        self.smoke_overlays: List[torch.Tensor] = self._load_overlays()

    def _load_overlays(self) -> List[torch.Tensor]:
        """Loads smoke images and ensures RGBA format."""
        if not self.overlay_path.exists():
            logger.warning(f"Smoke resource path not found: {self.overlay_path}")
            return []

        overlays = []
        valid_exts = {'.png', '.tiff', '.tif'} # Formats that support alpha

        for file_path in self.overlay_path.iterdir():
            if file_path.suffix.lower() in valid_exts:
                try:
                    # Force RGBA to ensure we have an Alpha channel
                    img = Image.open(file_path).convert('RGBA')
                    tensor = to_tensor(img) # converts to (C, H, W) in [0, 1]
                    overlays.append(tensor)
                except Exception as e:
                    logger.warning(f"Could not load smoke image {file_path.name}: {e}")
        
        if not overlays:
            logger.warning(f"No valid smoke images found in {self.overlay_path}")
            
        return overlays

    def extra_repr(self) -> str:
        return f"probability={self.probability}, loaded_overlays={len(self.smoke_overlays)}, opacity={self.opacity}"

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): Input image (C, H, W) or (N, C, H, W).
            
        Returns:
            torch.Tensor: Image with smoke overlay.
        """
        # 1. Early Exits
        if not self.smoke_overlays:
            return img
        
        if torch.rand(1, device=img.device) > self.probability:
            return img

        # 2. Setup Input
        # Handle Batch vs Single
        is_batch = img.ndim == 4
        x = img if is_batch else img.unsqueeze(0)
        
        N, C, H, W = x.shape
        device = x.device

        # Normalize Input to [0, 1] float for blending
        # Check if input is byte (0-255) or float (0-1)
        is_float_input = x.is_floating_point()
        needs_scale_down = False
        
        if not is_float_input or x.max() > 1.0:
            x_norm = x.float() / 255.0
            needs_scale_down = True
        else:
            x_norm = x

        # 3. Apply Smoke (Batch-wise or Single)
        # For performance, we pick ONE smoke pattern and apply it to the whole batch.
        # Alternatively, we could loop to pick a different one per image.
        
        smoke_idx = torch.randint(0, len(self.smoke_overlays), (1,)).item()
        smoke_tensor = self.smoke_overlays[smoke_idx].to(device)

        # Resize smoke to match input dimensions
        # smoke_tensor is (4, H_src, W_src) -> Resize to (4, H, W)
        smoke_resized = resize(smoke_tensor, size=(H, W), antialias=True)
        
        # Split Channels
        # Smoke RGB: (3, H, W)
        # Smoke Alpha: (1, H, W)
        s_rgb = smoke_resized[:3, :, :]
        s_alpha = smoke_resized[3:4, :, :] * self.opacity # Apply global opacity

        # 4. Alpha Blending Formula
        # Out = Src * Alpha + Dst * (1 - Alpha)
        #
        # x_norm might be (N, 3, H, W). s_rgb is (3, H, W).
        # Broadcasting handles the batch dimension N automatically.
        
        # Ensure input has 3 channels for blending (handle grayscale if needed)
        if C == 1:
            # If input is grayscale, we can either make output RGB or convert smoke to gray.
            # Let's convert smoke to gray for consistency: 0.299R + 0.587G + 0.114B
            s_rgb = (s_rgb[0] * 0.299 + s_rgb[1] * 0.587 + s_rgb[2] * 0.114).unsqueeze(0)
        elif C > 3:
            # Handle RGBA input: blend only on RGB, keep original Alpha?
            # For simplicity, we blend on first 3 channels.
            pass

        # Perform blending
        # Take only first 3 channels of input for calculation (ignore input alpha for now)
        input_rgb = x_norm[:, :3, :, :]
        
        blended_rgb = (s_rgb * s_alpha) + (input_rgb * (1.0 - s_alpha))
        
        # Reassemble output
        if C > 3:
            # Concatenate original alpha channel back
            out = torch.cat([blended_rgb, x_norm[:, 3:, :, :]], dim=1)
        elif C == 1:
             out = blended_rgb
        else:
             out = blended_rgb

        # 5. Restore Original Type/Range
        if needs_scale_down:
            out = (out * 255.0).clamp(0, 255)
            if not is_float_input:
                out = out.to(img.dtype)
        else:
            out = out.clamp(0, 1)

        return out if is_batch else out.squeeze(0)


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # 1. Create Dummy Smoke for Test (since file might not exist)
    # We subclass to override loading for the test
    class TestSmoke(Smoke):
        def _load_overlays(self):
            # Create a synthetic smoke puff (white circle with soft edges)
            H, W = 100, 100
            y, x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')
            dist = torch.sqrt(x*x + y*y)
            alpha = (1 - dist).clamp(0, 1) # Cone shape alpha
            
            # White smoke (RGB=1,1,1) + Alpha
            smoke = torch.ones(4, H, W)
            smoke[3] = alpha
            return [smoke]

    # 2. Setup
    # Black background to see white smoke
    img = torch.zeros(1, 3, 256, 256)
    
    # Augment
    aug = TestSmoke(probability=1.0, opacity=0.8)
    res = aug(img)

    # 3. Visualization
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    
    axs[0].imshow(img[0].permute(1, 2, 0))
    axs[0].set_title("Original (Black)")
    axs[0].axis('off')
    
    axs[1].imshow(res[0].permute(1, 2, 0))
    axs[1].set_title("Smoke Overlay")
    axs[1].axis('off')
    
    plt.tight_layout()
    plt.show()