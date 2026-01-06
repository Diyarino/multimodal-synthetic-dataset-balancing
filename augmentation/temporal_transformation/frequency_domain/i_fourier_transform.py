# -*- coding: utf-8 -*-
"""
Inverse Fourier Transform Module.

Converts signals or images from the Frequency Domain back to the Time/Spatial Domain.
Crucial for reconstructing signals after spectral manipulation.

@author: Diyar Altinses, M.Sc.
"""

import torch


class InverseFourierTransform(torch.nn.Module):
    """
    Applies the Inverse Fast Fourier Transform (IFFT) to the input.
    Expects complex-valued input tensors (Frequency Domain).
    
    - For 3D inputs (Batch, Channels, Length), applies 1D IFFT.
    - For 4D inputs (Batch, Channels, Height, Width), applies 2D IFFT.
    """

    def __init__(self, probability: float = 1.0, return_real: bool = True):
        """
        Parameters
        ----------
        probability : float
            Probability of applying the transform.
        return_real : bool
            If True, returns only the Real part of the result (discards imaginary errors).
            This is required if the output should be a standard image/signal.
            Default is True.
        """
        super().__init__()
        self.probability = probability
        self.return_real = return_real

    def extra_repr(self) -> str:
        return f"probability={self.probability}, return_real={self.return_real}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor (Complex).
                              Shape (N, C, L) or (N, C, H, W).
            
        Returns:
            torch.Tensor: Reconstructed Time/Spatial domain signal.
        """
        # 1. Probability Check
        if torch.rand(1, device=x.device) > self.probability:
            return x

        # 2. Determine Dimensions for IFFT
        # We assume the last dimension(s) are the frequency axes
        if x.ndim == 3:
            # 1D Signal
            out = torch.fft.ifft(x, dim=-1)
        elif x.ndim == 4:
            # 2D Image
            out = torch.fft.ifft2(x, dim=(-2, -1))
        else:
            # Fallback
            out = torch.fft.ifft(x)

        # 3. Handle Output Format
        if self.return_real:
            # Taking the real part is standard for reconstruction.
            # Theoretical inverse of a real-signal FFT should be real, 
            # but floating point errors create tiny imaginary parts.
            return out.real.float() 
        else:
            # Return full complex tensor if needed for further math
            return out


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # ---------------------------------------------------------
    # Test Workflow: Time -> Freq -> Time (Roundtrip)
    # ---------------------------------------------------------
    
    # 1. Create Original Signal
    t = torch.linspace(0, 1, 100)
    original = (torch.sin(20 * t) + 0.5 * torch.sin(50 * t)).view(1, 1, 100)
    
    # 2. Forward FFT (Simulation of previous step)
    freq_data = torch.fft.fft(original, dim=-1)
    
    # 3. Apply Inverse FFT (The Augmentation)
    aug_ifft = InverseFourierTransform(probability=1.0, return_real=True)
    reconstructed = aug_ifft(freq_data)

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    
    axs[0].plot(original[0, 0].numpy())
    axs[0].set_title("1. Original Signal")
    
    # Show Magnitude of Frequency Data
    axs[1].plot(torch.abs(freq_data[0, 0]).numpy())
    axs[1].set_title("2. Frequency Domain (FFT)\n(Input to IFourier)")
    
    axs[2].plot(reconstructed[0, 0].numpy(), linestyle='--')
    axs[2].set_title("3. Reconstructed (IFFT)\n(Output)")
    
    plt.tight_layout()
    plt.show()
    
    # Check Error
    err = (original - reconstructed).abs().max()
    print(f"Reconstruction Error: {err.item():.6f}")
    print("InverseFourierTransform test done.")