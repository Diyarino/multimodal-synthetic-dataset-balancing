# -*- coding: utf-8 -*-
"""
Spectral Phase Jitter Augmentation.

Adds random noise to the Phase of the signal in the Frequency Domain.
This corresponds to small translations or structural deformations in the Time/Spatial domain.

@author: Diyar Altinses, M.Sc.
"""

import torch
import torch.fft


class SpectralPhaseJitter(torch.nn.Module):
    """
    Randomly perturbs the phase of frequency components.
    
    Warning: The phase contains the structural information of a signal (edges in images, transients in audio).
    Modifying it too strongly can destroy the signal completely. Use small strength values!
    """

    def __init__(
        self, 
        probability: float = 1.0, 
        strength: float = 0.1, 
        fraction: float = 1.0
    ):
        """
        Parameters
        ----------
        probability : float
            Probability of applying the augmentation.
        strength : float
            Maximum phase shift in radians.
            Example: 0.1 means phase is shifted by random value in [-0.1, +0.1] rad.
            Pi (3.14) would be a complete scrambling.
        fraction : float
            Percentage of frequency bins to perturb (0.0 to 1.0).
            Default 1.0 (perturb all phases slightly).
        """
        super().__init__()
        self.probability = probability
        self.strength = strength
        self.fraction = fraction

    def extra_repr(self) -> str:
        return f"probability={self.probability}, strength={self.strength}, fraction={self.fraction}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor (N, C, L) or (N, C, H, W).
            
        Returns:
            torch.Tensor: Augmented tensor.
        """
        # 1. Probability Check
        if torch.rand(1, device=x.device) > self.probability:
            return x

        # 2. Determine Dimensions
        if x.ndim == 3:
            fft_dims = (-1,)
        elif x.ndim == 4:
            fft_dims = (-2, -1)
        else:
            return x

        # 3. FFT (Real input -> Complex spectrum)
        x_freq = torch.fft.rfftn(x, dim=fft_dims)
        
        mag = torch.abs(x_freq)
        phase = torch.angle(x_freq)

        # 4. Generate Additive Phase Noise (Vectorized)
        # We want to add noise, not multiply. 
        # Multiplication breaks phase wrapping continuity (-pi to pi).
        
        # Create mask for selected frequencies
        if self.fraction < 1.0:
            mask = torch.rand_like(phase) < self.fraction
        else:
            mask = True

        # Generate noise in range [-strength, +strength]
        # (rand - 0.5) * 2 -> [-1, 1]
        noise = (torch.rand_like(phase) - 0.5) * 2 * self.strength
        
        # Apply noise only where mask is True
        if isinstance(mask, torch.Tensor):
            noise = noise * mask

        # Add noise to original phase
        phase_modified = phase + noise

        # 5. Reconstruct
        # No need to manually wrap phase to [-pi, pi], torch.polar handles angle modulo automatically.
        x_modified = torch.polar(mag, phase_modified)
        
        out = torch.fft.irfftn(x_modified, s=x.shape[fft_dims[0]:], dim=fft_dims)

        return out


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # ---------------------------------------------------------
    # Test 1: 1D Pulse Signal (Sensitive to Phase)
    # ---------------------------------------------------------
    # A pulse (delta) relies heavily on all phases aligning perfectly at one point.
    L = 200
    signal = torch.zeros(1, 1, L)
    signal[0, 0, 100] = 1.0 # Delta function in middle
    
    # Augment: Perturb phase slightly (0.5 rad)
    # This should smear the sharp pulse out in time (Dispersion)
    aug = SpectralPhaseJitter(probability=1.0, strength=0.5, fraction=1.0)
    out = aug(signal.clone())

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    
    # Time Domain Plot
    axs[0].plot(signal[0, 0].numpy(), label="Original Pulse")
    axs[0].plot(out[0, 0].numpy(), label="Phase Jittered", alpha=0.8)
    axs[0].set_title("Time Domain (Pulse Dispersion)")
    axs[0].legend()
    
    # Frequency Magnitude Check (Should be identical!)
    # Phase augmentation should NOT change the magnitude spectrum (energy distribution)
    fft_orig = torch.fft.rfft(signal).abs().flatten()
    fft_aug = torch.fft.rfft(out).abs().flatten()
    
    # We plot difference to prove magnitude is preserved
    diff = (fft_orig - fft_aug).abs()
    axs[1].plot(diff.numpy())
    axs[1].set_ylim(-0.1, 0.1)
    axs[1].set_title("Magnitude Difference\n(Should be near zero)")
    
    plt.tight_layout()
    plt.show()
    print("SpectralPhaseJitter test done.")