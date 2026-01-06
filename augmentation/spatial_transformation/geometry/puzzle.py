# -*- coding: utf-8 -*-
"""
2D Puzzle / Patch Shuffle Augmentation.

Splits an image into a grid of patches and shuffles them randomly.
Preserves local structure within patches but destroys global structure.
High-performance implementation using tensor views.

@author: Diyar Altinses, M.Sc.
"""

from typing import Tuple
import torch


class Puzzle2D(torch.nn.Module):
    """
    Divides the image into a grid (e.g., 3x3) and randomly shuffles the grid cells.
    Margins that don't fit the grid are preserved (not shuffled).
    """

    def __init__(self, probability: float = 1.0, pieces: Tuple[int, int] = (3, 3)):
        """
        Parameters
        ----------
        probability : float
            Probability of applying the shuffle.
        pieces : tuple
            (rows, cols) to split the image into.
            Example: (3, 3) creates 9 patches.
        """
        super().__init__()
        self.probability = probability
        self.grid_size = pieces # (rows, cols)

    def extra_repr(self) -> str:
        return f"probability={self.probability}, grid={self.grid_size}"

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img (torch.Tensor): Input image (C, H, W) or (N, C, H, W).
            
        Returns:
            torch.Tensor: Shuffled image.
        """
        # 1. Probability Check
        if torch.rand(1, device=img.device) > self.probability:
            return img

        # 2. Setup Dimensions
        is_batch = img.ndim == 4
        x = img if is_batch else img.unsqueeze(0)
        N, C, H, W = x.shape
        
        rows, cols = self.grid_size

        # 3. Handle Remainders
        # If dimensions aren't perfectly divisible, we process a cropped center/top-left
        # and paste it back later.
        patch_h = H // rows
        patch_w = W // cols
        
        if patch_h == 0 or patch_w == 0:
            return img  # Image too small for grid

        # Effective area that can be puzzled
        eff_h = patch_h * rows
        eff_w = patch_w * cols
        
        # Clone is needed because we modify parts of it (or overwrite output)
        out = x.clone()

        # Extract the area to be shuffled
        # Shape: (N, C, eff_h, eff_w)
        roi = x[..., :eff_h, :eff_w]

        # 4. Reshape into Patches (The "View" Trick)
        # We want to view the tensor as a grid of patches.
        # Step A: View as (N, C, rows, patch_h, cols, patch_w)
        # Step B: Permute to (N, rows, cols, C, patch_h, patch_w) to group grid cells
        # Step C: Flatten grid cells into one dimension -> (N, num_patches, C, patch_h, patch_w)
        
        num_patches = rows * cols
        
        patches = roi.view(N, C, rows, patch_h, cols, patch_w)
        patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        patches_flat = patches.view(N, num_patches, C, patch_h, patch_w)

        # 5. Generate Random Permutations
        # We need independent permutations for each image in the batch.
        # rand() -> argsort() gives random indices
        rand_scores = torch.rand(N, num_patches, device=x.device)
        perm_indices = torch.argsort(rand_scores, dim=1) # Shape (N, num_patches)

        # 6. Apply Shuffle via Gather
        # Expand indices to match patch dimensions for gathering
        # Shape need: (N, num_patches, C, patch_h, patch_w)
        indices_expanded = perm_indices.view(N, num_patches, 1, 1, 1).expand(-1, -1, C, patch_h, patch_w)
        
        shuffled_patches = torch.gather(patches_flat, 1, indices_expanded)

        # 7. Reassemble Image
        # Reverse the reshaping process
        # Shape: (N, rows, cols, C, patch_h, patch_w)
        shuffled_grid = shuffled_patches.view(N, rows, cols, C, patch_h, patch_w)
        
        # Permute back to spatial layout: (N, C, rows, patch_h, cols, patch_w)
        restored_layout = shuffled_grid.permute(0, 3, 1, 4, 2, 5).contiguous()
        
        # Merge spatial dims: (N, C, eff_h, eff_w)
        shuffled_roi = restored_layout.view(N, C, eff_h, eff_w)

        # 8. Paste back into output (preserves original margins/remainder)
        out[..., :eff_h, :eff_w] = shuffled_roi

        return out if is_batch else out.squeeze(0)


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # Create an image with a clear structure (Gradient + Grid)
    # Shape: (1, 3, 256, 256)
    H, W = 256, 256
    img = torch.zeros(1, 3, H, W)
    
    # Create a gradient
    y, x = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing='ij')
    img[0, 0] = x  # Red channel increases horizontally
    img[0, 1] = y  # Green channel increases vertically
    
    # Add a white cross to see position easily
    img[0, :, H//2-5:H//2+5, :] = 1.0
    img[0, :, :, W//2-5:W//2+5] = 1.0
    
    # Augment: 3x3 Grid
    aug = Puzzle2D(probability=1.0, pieces=(3, 3))
    res = aug(img)

    # Visualization
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    
    axs[0].imshow(img[0].permute(1, 2, 0))
    axs[0].set_title("Original")
    axs[0].axis('off')
    
    axs[1].imshow(res[0].permute(1, 2, 0))
    axs[1].set_title("Puzzle 3x3 (Shuffled)")
    axs[1].axis('off')
    
    # Draw grid lines on the result to show where cuts happened
    # Just for visualization
    for i in range(1, 3):
        axs[1].axvline(x=i * (W // 3), color='white', linestyle='--', linewidth=0.5)
        axs[1].axhline(y=i * (H // 3), color='white', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()
    print("Puzzle2D test done.")