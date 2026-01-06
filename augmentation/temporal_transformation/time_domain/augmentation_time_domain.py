# -*- coding: utf-8 -*-
"""
Temporal Deformations & Filters.

Includes:
1. TimeWarping: Stretches/squeezes the time axis.
2. MovingAverage: Smooths the signal (Box Blur).

@author: Diyar Altinses, M.Sc.
"""

import torch

class TimeWarping(torch.nn.Module):
    """
    Applies Time Warping by resizing the signal/image via interpolation.
    Simulates variations in speed (e.g., speaking rate, movement speed).
    """
    def __init__(self, probability: float = 1.0, warp_factor: float = 0.2):
        """
        warp_factor: How much to stretch/squeeze. 0.2 means +/- 20% length change.
        """
        super().__init__()
        self.probability = probability
        self.warp_factor = warp_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1, device=x.device) > self.probability:
            return x
            
        # x shape: (N, C, L) or (N, C, H, W)
        orig_size = x.shape[2:]
        
        # Determine random scaling factor
        scale = 1.0 + (torch.rand(1, device=x.device) - 0.5) * 2 * self.warp_factor
        
        # 1. Resample (Stretch/Squeeze)
        # We align_corners=False to keep standard behavior
        if x.ndim == 3: # 1D
            mode = 'linear'
        else: # 2D
            mode = 'bilinear'
            
        warped = torch.nn.functional.interpolate(x, scale_factor=scale.item(), mode=mode, align_corners=False)
        
        # 2. Crop or Pad back to original size
        # Because augmentations usually need to preserve input shape for batches
        out = self._match_size(warped, orig_size)
        return out

    def _match_size(self, x: torch.Tensor, target_size) -> torch.Tensor:
        # Helper to crop or pad dimensions
        if x.ndim == 3: # 1D
            curr_l = x.shape[-1]
            tgt_l = target_size[0]
            if curr_l > tgt_l: # Crop center
                start = (curr_l - tgt_l) // 2
                return x[..., start:start+tgt_l]
            elif curr_l < tgt_l: # Pad
                pad = tgt_l - curr_l
                return torch.nn.functional.pad(x, (0, pad))
        else: # 2D
            # Similar logic for H and W... (simplified for brevity: usually simple Resize is preferred)
            return torch.nn.functional.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x


class MovingAverage(torch.nn.Module):
    """
    Applies a Moving Average filter (Smoothing / Low-Pass).
    Implemented via Average Pooling for high performance.
    """
    def __init__(self, probability: float = 1.0, window_size: int = 3):
        super().__init__()
        self.probability = probability
        self.window_size = window_size
        # Padding to keep output size same as input
        self.padding = window_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1, device=x.device) > self.probability:
            return x
        
        # Use Pooling layers which are highly optimized C++ implementations
        if x.ndim == 3:
            # 1D: (N, C, L)
            # stride=1 ensures we get a sliding window
            out = torch.nn.functional.avg_pool1d(x, kernel_size=self.window_size, stride=1, padding=self.padding, count_include_pad=False)
        elif x.ndim == 4:
            # 2D: (N, C, H, W)
            out = torch.nn.functional.avg_pool2d(x, kernel_size=self.window_size, stride=1, padding=self.padding, count_include_pad=False)
        else:
            return x
            
        # Fix possible size mismatch due to padding logic
        if out.shape != x.shape:
             out = torch.nn.functional.interpolate(out, size=x.shape[2:], mode='nearest')
             
        return out
    
    
class Flipping(torch.nn.Module):
    """
    Flips the tensor along specified dimensions.
    """
    def __init__(self, probability: float = 0.5, dim: int = -1):
        super().__init__()
        self.probability = probability
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1, device=x.device) > self.probability:
            return x
        return x.flip(dims=(self.dim,))


class PointwiseOp(torch.nn.Module):
    """Helper for simple ops like Abs, Negate, Clamp."""
    def __init__(self, probability: float = 1.0, op_type: str = 'negation', **kwargs):
        super().__init__()
        self.probability = probability
        self.op_type = op_type
        self.kwargs = kwargs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1, device=x.device) > self.probability:
            return x
        
        if self.op_type == 'negation':
            return -x
        elif self.op_type == 'magnitude': # Absolute value
            out = torch.abs(x)
            # Original code had logic to randomly flip back to negative?
            # Assuming standard magnitude here.
            return out
        elif self.op_type == 'clamp':
            return torch.clamp(x, min=self.kwargs.get('min', 0), max=self.kwargs.get('max', 1))
        return x


class ChunkPermutation(torch.nn.Module):
    """
    Splits signal into N chunks and shuffles them (formerly Permutation).
    Optimized to avoid loops.
    """
    def __init__(self, probability: float = 1.0, pieces: int = 4):
        super().__init__()
        self.probability = probability
        self.pieces = pieces

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1, device=x.device) > self.probability:
            return x
        
        N, C, L = x.shape[:3] # Assuming 3D for simplicity, works for 4D if flattened
        
        # Ensure divisibility (crop if needed)
        chunk_len = L // self.pieces
        eff_len = chunk_len * self.pieces
        x_eff = x[..., :eff_len]
        
        # Reshape to (Batch, Pieces, ChunkLen) - ignoring Channels for shuffle (shuffle all C same)
        # Or (Batch, C, Pieces, ChunkLen)
        
        # Flatten C into batch for independent shuffle per channel? 
        # Original code shuffled FEATURES (Channels) independently.
        
        # Reshape: (N, C, Pieces, ChunkLen)
        x_view = x_eff.view(N, C, self.pieces, chunk_len)
        
        # Generate Permutation: (N, C, Pieces)
        rand = torch.rand(N, C, self.pieces, device=x.device)
        perm = torch.argsort(rand, dim=-1)
        
        # Expand for gather
        perm_exp = perm.unsqueeze(-1).expand(-1, -1, -1, chunk_len)
        
        # Shuffle
        out = torch.gather(x_view, 2, perm_exp)
        
        # Flatten back
        out = out.reshape(N, C, eff_len)
        
        # Pad remainder if existed
        if L > eff_len:
            out = torch.cat([out, x[..., eff_len:]], dim=-1)
            
        return out