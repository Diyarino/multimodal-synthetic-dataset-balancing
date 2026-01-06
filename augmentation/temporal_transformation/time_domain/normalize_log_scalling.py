# -*- coding: utf-8 -*-
"""
Logarithmic Scaling Augmentation.

Compresses the dynamic range of the signal.
Useful for data with exponential distributions (e.g. audio amplitude, financial data).

@author: Diyar Altinses, M.Sc.
"""

import torch

class LogScale(torch.nn.Module):
    """
    Applies logarithmic scaling to the input.
    Formula: sign(x) * log(|x| + 1)  or  log(x + eps)
    
    This compresses large values while preserving small details.
    """

    def __init__(
        self, 
        probability: float = 1.0, 
        base: str = 'e', 
        keep_sign: bool = True,
        eps: float = 1e-6
    ):
        """
        Parameters
        ----------
        probability : float
            Probability of applying the augmentation.
        base : str
            'e' for natural log (torch.log), '10' for log10, '2' for log2.
        keep_sign : bool
            If True, preserves the sign of negative inputs: sign(x) * log(|x| + 1).
            If False, assumes non-negative inputs or takes abs: log(|x| + eps).
        eps : float
            Small constant to prevent log(0) -> -inf.
        """
        super().__init__()
        self.probability = probability
        self.base = base
        self.keep_sign = keep_sign
        self.eps = eps

    def extra_repr(self) -> str:
        return f"probability={self.probability}, base={self.base}, keep_sign={self.keep_sign}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor.
        """
        # 1. Global Probability Check (Fast exit)
        if self.probability == 0.0:
            return x
            
        N = x.shape[0]
        device = x.device

        # 2. Determine per-sample mask
        # Shape (N,)
        do_aug = torch.rand(N, device=device) < self.probability
        
        if not do_aug.any():
            return x

        # 3. Prepare Mask for broadcasting
        # (N, 1, 1...)
        view_shape = [N] + [1] * (x.ndim - 1)
        mask = do_aug.view(*view_shape).float()

        # 4. Compute Log
        # Safe log computation handling zeros and negatives
        if self.keep_sign:
            # Method: sign(x) * log(|x| + 1)
            # Adding 1 ensures that 0 maps to 0 and we don't get negative logs for small values
            x_abs = x.abs()
            x_log = torch.log1p(x_abs) # log(1 + x) is more accurate for small x
        else:
            # Method: log(|x| + eps)
            x_abs = x.abs()
            x_log = torch.log(x_abs + self.eps)

        # Handle Base
        if self.base == '10':
            x_log = x_log / 2.302585 # divide by ln(10)
        elif self.base == '2':
            x_log = x_log / 0.693147 # divide by ln(2)
            
        # Re-apply sign if requested
        if self.keep_sign:
            x_log = x_log * torch.sign(x)

        # 5. Blend Original and Augmented
        # Output = Mask * Log + (1-Mask) * Original
        return mask * x_log + (1.0 - mask) * x


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # ---------------------------------------------------------
    # Test Data: Exponential Growth + Negative Values
    # ---------------------------------------------------------
    t = torch.linspace(0, 5, 100)
    # Signal grows exponentially: e^t
    # We flip half to negative to test sign handling
    signal = torch.exp(t).view(1, 1, 100)
    signal[0, 0, 50:] *= -1 
    
    # 1. Natural Log (Keep Sign) -> Should linearize the exponential growth
    aug_ln = LogScale(probability=1.0, base='e', keep_sign=True)
    out_ln = aug_ln(signal.clone())
    
    # 2. Log10 (Magnitude only)
    aug_log10 = LogScale(probability=1.0, base='10', keep_sign=False)
    out_log10 = aug_log10(signal.clone())

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------
    
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    
    axs[0].plot(signal[0, 0].numpy())
    axs[0].set_title("Original (Exponential Growth)")
    
    axs[1].plot(out_ln[0, 0].numpy())
    axs[1].set_title("Log Scale (Natural, Keep Sign)\nResult is Linear!")
    
    axs[2].plot(out_log10[0, 0].numpy())
    axs[2].set_title("Log10 Scale (Magnitude Only)\nValues become small positive numbers")
    
    plt.tight_layout()
    plt.show()
    print("LogScale test done.")