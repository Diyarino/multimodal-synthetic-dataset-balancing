# -*- coding: utf-8 -*-
"""
Permutation & Shuffling Augmentations.

Rearranges data elements without changing their values.
1. RandomChannelShuffle: Swaps channels (e.g. RGB -> GBR).
2. RandomChunkShuffle: Splits time series into segments and shuffles them (Jigsaw).

@author: Diyar Altinses, M.Sc.
"""

import torch


class RandomChannelShuffle(torch.nn.Module):
    """
    Randomly permutes the channel order.
    Useful to make models invariant to sensor order or color channel order.
    (Formerly PermuteFeatures / PermuteChannels).
    """

    def __init__(self, probability: float = 0.5):
        super().__init__()
        self.probability = probability

    def extra_repr(self) -> str:
        return f"probability={self.probability}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input (N, C, L) or (N, C, H, W).
        """
        if torch.rand(1, device=x.device) > self.probability:
            return x

        N, C = x.shape[:2]
        
        # We generate a random permutation for EACH sample in the batch independently.
        # Generating N permutations of size C efficiently:
        
        # 1. Generate random scores: (N, C)
        rand_scores = torch.rand(N, C, device=x.device)
        
        # 2. Get indices that would sort these scores: (N, C)
        # This gives us N random permutations of indices [0..C-1]
        perm_indices = torch.argsort(rand_scores, dim=1)
        
        # 3. Expand indices for gathering
        # We need to match the dimensionality of x to use torch.gather
        # x: (N, C, L, ...) -> indices: (N, C, 1, ...)
        
        # View shape: (N, C) -> (N, C, 1, 1...)
        view_shape = [N, C] + [1] * (x.ndim - 2)
        expand_shape = list(x.shape)
        
        perm_indices = perm_indices.view(*view_shape).expand(*expand_shape)
        
        # 4. Gather
        # Gather along dimension 1 (Channels)
        return torch.gather(x, 1, perm_indices)


class RandomChunkShuffle(torch.nn.Module):
    """
    Splits the signal into 'pieces' and shuffles them randomly.
    (Time Domain Jigsaw Puzzle).
    
    Formerly 'Permutation'.
    """

    def __init__(self, probability: float = 1.0, pieces: int = 4, same_across_channels: bool = True):
        """
        Parameters
        ----------
        probability : float
            Probability.
        pieces : int
            Number of segments to split the signal into.
        same_across_channels : bool
            If True, all channels are shuffled in the same order (preserves inter-channel timing).
            If False, each channel is shuffled differently (destroys inter-channel correlations).
        """
        super().__init__()
        self.probability = probability
        self.pieces = pieces
        self.same_across_channels = same_across_channels

    def extra_repr(self) -> str:
        return f"prob={self.probability}, pieces={self.pieces}, synced={self.same_across_channels}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input (N, C, L).
        """
        if torch.rand(1, device=x.device) > self.probability:
            return x

        N, C, L = x.shape
        device = x.device
        
        # 1. Check Divisibility
        chunk_len = L // self.pieces
        eff_len = chunk_len * self.pieces
        
        # Cut off remainder (or handle it, here we just crop end)
        x_core = x[..., :eff_len]
        remainder = x[..., eff_len:]
        
        # 2. Reshape into Chunks
        # Shape: (N, C, Pieces, ChunkLen)
        x_view = x_core.view(N, C, self.pieces, chunk_len)
        
        # 3. Generate Permutations
        # If synced: One perm per sample (N, 1, Pieces)
        # If not synced: One perm per channel (N, C, Pieces)
        if self.same_across_channels:
            rand_scores = torch.rand(N, 1, self.pieces, device=device)
        else:
            rand_scores = torch.rand(N, C, self.pieces, device=device)
            
        # Indices: (N, 1/C, Pieces)
        perm_indices = torch.argsort(rand_scores, dim=-1)
        
        # 4. Expand for Gather
        # We need indices to match (N, C, Pieces, ChunkLen)
        # Current indices: (N, 1/C, Pieces) -> Unsqueeze last dim -> (N, 1/C, Pieces, 1)
        # Expand -> (N, C, Pieces, ChunkLen)
        perm_indices = perm_indices.unsqueeze(-1).expand(N, C, self.pieces, chunk_len)
        
        # 5. Gather (Shuffle)
        # Shuffle along dimension 2 (Pieces)
        shuffled = torch.gather(x_view, 2, perm_indices)
        
        # 6. Flatten back
        out = shuffled.reshape(N, C, eff_len)
        
        # Append remainder
        if remainder.shape[-1] > 0:
            out = torch.cat([out, remainder], dim=-1)
            
        return out


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # ---------------------------------------------------------
    # Test Data: Ramp Signal (0 to 100)
    # Shape: (1, 2, 100) -> 2 Channels
    # ---------------------------------------------------------
    L = 100
    ramp = torch.linspace(0, 10, L)
    data = torch.stack([ramp, ramp + 5], dim=0).unsqueeze(0) # (1, 2, 100)
    
    # 1. Channel Shuffle
    # With 2 channels, this just swaps them or keeps them.
    aug_ch = RandomChannelShuffle(probability=1.0)
    out_ch = aug_ch(data.clone())
    
    # 2. Chunk Shuffle (Synced)
    # Both lines should be broken in the same way
    aug_chunk_sync = RandomChunkShuffle(probability=1.0, pieces=5, same_across_channels=True)
    out_chunk_sync = aug_chunk_sync(data.clone())

    # 3. Chunk Shuffle (Independent)
    # Lines should be broken differently
    aug_chunk_indep = RandomChunkShuffle(probability=1.0, pieces=5, same_across_channels=False)
    out_chunk_indep = aug_chunk_indep(data.clone())

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------

    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    
    # Plot Chunk Shuffle Synced
    axs[0].plot(out_chunk_sync[0, 0].numpy(), label="Ch 0")
    axs[0].plot(out_chunk_sync[0, 1].numpy(), label="Ch 1")
    axs[0].set_title("Chunk Shuffle (Synced)\nChannels move together")
    axs[0].legend()
    
    # Plot Chunk Shuffle Independent
    axs[1].plot(out_chunk_indep[0, 0].numpy(), label="Ch 0")
    axs[1].plot(out_chunk_indep[0, 1].numpy(), label="Ch 1")
    axs[1].set_title("Chunk Shuffle (Independent)\nChannels disconnected")
    axs[1].legend()
    
    # Original
    axs[2].plot(data[0, 0].numpy(), 'k--', alpha=0.5, label="Orig Ch 0")
    axs[2].plot(data[0, 1].numpy(), 'k:', alpha=0.5, label="Orig Ch 1")
    axs[2].set_title("Original Ramps")
    axs[2].legend()
    
    plt.tight_layout()
    plt.show()
    print("Permutation tests done.")