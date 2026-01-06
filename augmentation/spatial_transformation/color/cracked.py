# -*- coding: utf-8 -*-
"""
Cracked Augmentation Module.

Simulates a broken lens/glass effect by overlaying transparent crack images.
Designed for PyTorch pipelines.

@author: Diyar Altinses, M.Sc.
"""

import logging
from pathlib import Path
from typing import Optional, List, Union

import torch
from torchvision.transforms.functional import resize, to_tensor
from PIL import Image

# %% Logging Setup

logger = logging.getLogger(__name__)

# %%

class Cracked(torch.nn.Module):
    """
    Simulates a broken lens effect by overlaying a crack image with alpha blending.
    
    Attributes:
        probability (float): Probability of applying the transform.
        overlay_path (Path): Directory containing the crack overlay images (PNG with Alpha).
        output_dtype (torch.dtype): Desired output data type.
    """

    def __init__(
        self, 
        probability: float = 0.5, 
        path: Optional[Union[str, Path]] = None, 
        output_dtype: torch.dtype = torch.float32
    ):
        """
        Args:
            probability (float): Probability of applying the effect (0.0 - 1.0).
            path (str or Path, optional): Path to folder with crack PNGs. 
                                          Defaults to '../resources/cracked'.
            output_dtype (torch.dtype): Target dtype (e.g., torch.float32, torch.uint8).
        """
        super().__init__()
        self.probability = probability
        self.output_dtype = output_dtype
        
        # Determine Path
        if path is None:
            # Resolves robustly relative to this file's location
            self.overlay_path = Path(__file__).resolve().parent.parent / 'resources' / 'cracked'
        else:
            self.overlay_path = Path(path)

        self.cracks: List[torch.Tensor] = self._load_overlays()

    def _load_overlays(self) -> List[torch.Tensor]:
        """Loads and caches overlay images into memory."""
        if not self.overlay_path.exists():
            logger.warning(f"Overlay path not found: {self.overlay_path}. Effect will be disabled.")
            return []

        overlays = []
        # Support common image formats
        valid_extensions = {'.png', '.jpg', '.jpeg', '.tiff'}
        
        for file_path in self.overlay_path.iterdir():
            if file_path.suffix.lower() in valid_extensions:
                try:
                    # Load and ensure RGBA (4 channels)
                    img = Image.open(file_path).convert('RGBA')
                    overlays.append(to_tensor(img))
                except Exception as e:
                    logger.warning(f"Could not load overlay {file_path.name}: {e}")
        
        if not overlays:
            logger.warning(f"No valid images found in {self.overlay_path}")
            
        return overlays

    def extra_repr(self) -> str:
        return f"probability={self.probability}, overlays_loaded={len(self.cracks)}, dtype={self.output_dtype}"

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Applies the crack effect.

        Args:
            img (torch.Tensor): Input image. Shape (C, H, W) or (N, C, H, W).
                                Assumes standard layout (Channels first).

        Returns:
            torch.Tensor: Augmented image.
        """
        if not self.cracks:
            return img

        # 1. Probability Check
        if torch.rand(1, device=img.device) > self.probability:
            return img

        # 2. Input normalization (handle 0-255 vs 0-1)
        # Avoid modifying original tensor in-place
        x = img.clone()
        
        # Simple heuristic: if max > 1, assume 0-255 range
        needs_normalization = x.max() > 1.0
        if needs_normalization:
            x = x / 255.0

        # Ensure we work with first 3 channels (RGB) if input is RGBA
        # Keeping reference to Alpha if it exists in input could be tricky, 
        # usually lens cracks affect the visible RGB part.
        original_channels = x.shape[-3]
        if original_channels > 3:
            x_rgb = x[..., :3, :, :]
        else:
            x_rgb = x

        # 3. Select and Prepare Overlay
        crack_idx = torch.randint(0, len(self.cracks), (1,)).item()
        crack_tensor = self.cracks[crack_idx].to(img.device)

        # Resize overlay to match input spatial dimensions (H, W)
        target_size = x.shape[-2:] 
        crack_resized = resize(crack_tensor, size=target_size, antialias=True)

        # 4. Alpha Blending
        # Crack Tensor is (4, H, W): RGB + Alpha
        overlay_rgb = crack_resized[:3, :, :]
        overlay_alpha = crack_resized[3:4, :, :]  # Keep dim for broadcasting

        # Formula: Out = (Overlay * Alpha) + (Background * (1 - Alpha))
        # Broadcasting handles Batch (N, C, H, W) automatically against (3, H, W) and (1, H, W)
        blended = (overlay_rgb * overlay_alpha) + (x_rgb * (1.0 - overlay_alpha))

        # 5. Restore Structure
        if original_channels > 3:
            # Concatenate original alpha channel back if it existed
            out = torch.cat([blended, x[..., 3:, :, :]], dim=-3)
        else:
            out = blended

        # 6. Restore Range and Type
        if needs_normalization:
            out = out * 255.0
            
        return out.to(dtype=self.output_dtype)

# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # Setup standard logging for the test
    logging.basicConfig(level=logging.INFO)

    # Mock Data: Batch of 2 images, 3 Channels, 256x256
    image_batch = torch.ones(2, 3, 256, 256) 
    
    # Init Augmentation
    # Note: Ensure you have a valid path or create a dummy folder for testing
    try:
        augmentation = Cracked(probability=1.0, output_dtype=torch.float32)
        
        # Simulate loading if folder is empty/missing just for this test script to run without error
        if not augmentation.cracks:
            print("INFO: No crack images found. Creating dummy overlay for demo.")
            dummy_crack = torch.zeros(4, 100, 100)
            dummy_crack[0, :, :] = 1.0 # Red
            dummy_crack[3, 40:60, :] = 1.0 # Alpha strip in middle
            augmentation.cracks = [dummy_crack]

        augmented_batch = augmentation(image_batch)

        # Visualization
        f, axs = plt.subplots(1, 2, figsize=(8, 4))
        
        axs[0].set_title("Original Img 1")
        axs[0].imshow(image_batch[0].permute(1, 2, 0))

        axs[1].set_title("Cracked Img 1")
        axs[1].imshow(augmented_batch[0].permute(1, 2, 0))
        
        
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Test failed: {e}")