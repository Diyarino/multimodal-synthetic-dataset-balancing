# -*- coding: utf-8 -*-
"""
Signal Event Injection (Gaussian & Step).

Simulates transient anomalies and sensor faults.
1. GaussianEvent: Adds a temporary spike/bump (e.g., impact, interference).
2. StepEvent: Adds a sudden baseline shift (e.g., sensor calibration loss).

@author: Diyar Altinses, M.Sc.
"""

import torch


class GaussianEvent(torch.nn.Module):
    """
    Adds a Gaussian bump (bell curve) to random channels.
    Simulates transient spikes or temporary interference.
    
    Formula: x + gain * exp(- (t - mu)^2 / (2 * sigma^2))
    """

    def __init__(
        self, 
        probability: float = 1.0, 
        gain_range: tuple = (0.5, 2.0), 
        width_range: tuple = (0.02, 0.1),
        injection_prob: float = 0.5
    ):
        """
        Parameters
        ----------
        probability : float
            Global probability of applying the augmentation to the batch.
        gain_range : tuple
            Min and max height (amplitude) of the peak.
        width_range : tuple
            Min and max width (standard deviation) of the peak as a fraction of signal length.
            (0.02 = very sharp spike, 0.1 = wide bump).
        injection_prob : float
            Probability per channel to receive an event (0.0 to 1.0).
        """
        super().__init__()
        self.probability = probability
        self.gain_range = gain_range
        self.width_range = width_range
        self.injection_prob = injection_prob

    def extra_repr(self) -> str:
        return f"prob={self.probability}, gain={self.gain_range}, width={self.width_range}, inject_p={self.injection_prob}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor (N, C, L).
            
        Returns:
            torch.Tensor: Augmented tensor.
        """
        # 1. Global Probability Check
        if torch.rand(1, device=x.device) > self.probability:
            return x

        N, C, L = x.shape
        device = x.device

        # 2. Determine which channels get an event
        # Mask: (N, C, 1)
        # 1.0 if event should be added, 0.0 otherwise
        mask = (torch.rand(N, C, 1, device=device) < self.injection_prob).float()
        
        # Optimization: If no events are injected, return early
        if mask.sum() == 0:
            return x

        # 3. Generate Random Parameters (Vectorized)
        # Positions (mu): Random point in time [0, 1]
        mu = torch.rand(N, C, 1, device=device)
        
        # Widths (sigma): Random width in range
        sigma = torch.empty(N, C, 1, device=device).uniform_(*self.width_range)
        
        # Gains (amplitude): Random gain
        gain = torch.empty(N, C, 1, device=device).uniform_(*self.gain_range)
        
        # 4. Create Time Grid
        # Shape (1, 1, L) -> Broadcastable to (N, C, L)
        # Normalized time [0, 1] to match mu/sigma
        t = torch.linspace(0, 1, L, device=device).view(1, 1, L)
        
        # 5. Calculate Gaussian Curve
        # Formula: gain * exp( -0.5 * ((t - mu) / sigma)^2 )
        # Shapes: (1,1,L) - (N,C,1) -> (N,C,L)
        gauss = gain * torch.exp(-0.5 * ((t - mu) / sigma) ** 2)
        
        # 6. Apply
        # Add gaussian only where mask is 1
        return x + (gauss * mask)


class StepEvent(torch.nn.Module):
    """
    Adds a Step Function (Heaviside) to random channels.
    Simulates a sudden, permanent shift in sensor baseline (bias drift).
    
    Formula: x + gain * (t >= start_time)
    """

    def __init__(
        self, 
        probability: float = 1.0, 
        gain_range: tuple = (-1.0, 1.0),
        injection_prob: float = 0.5
    ):
        """
        Parameters
        ----------
        probability : float
            Global probability.
        gain_range : tuple
            Range of the step height (can be negative for downward steps).
        injection_prob : float
            Probability per channel to receive a step.
        """
        super().__init__()
        self.probability = probability
        self.gain_range = gain_range
        self.injection_prob = injection_prob

    def extra_repr(self) -> str:
        return f"prob={self.probability}, gain={self.gain_range}, inject_p={self.injection_prob}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Global Probability Check
        if torch.rand(1, device=x.device) > self.probability:
            return x

        N, C, L = x.shape
        device = x.device

        # 2. Determine active channels
        mask = (torch.rand(N, C, 1, device=device) < self.injection_prob).float()
        if mask.sum() == 0:
            return x

        # 3. Generate Parameters
        # Start Index: Random integer [0, L-1]
        start_indices = torch.randint(0, L, (N, C, 1), device=device)
        
        # Gain: Random float
        gain = torch.empty(N, C, 1, device=device).uniform_(*self.gain_range)

        # 4. Create Step Mask (Vectorized)
        # Grid: (1, 1, L) containing [0, 1, 2, ..., L-1]
        grid = torch.arange(L, device=device).view(1, 1, L)
        
        # Step is 1.0 where grid >= start_index
        # Broadcasting: (1,1,L) >= (N,C,1) -> (N,C,L)
        step_shape = (grid >= start_indices).float()
        
        # 5. Apply
        # x + (StepShape * Gain * ChannelMask)
        return x + (step_shape * gain * mask)


# %% Test Block

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # ---------------------------------------------------------
    # Test Data: Flat lines (Zero)
    # Shape: (Batch=3, Channels=1, Length=100)
    # ---------------------------------------------------------
    L = 100
    signal = torch.zeros(3, 1, L)
    
    # ---------------------------------------------------------
    # 1. Gaussian Event Test
    # ---------------------------------------------------------
    # Add peaks with height 0.5 to 1.0
    aug_gauss = GaussianEvent(probability=1.0, gain_range=(0.5, 1.0), injection_prob=1.0)
    out_gauss = aug_gauss(signal.clone())

    # ---------------------------------------------------------
    # 2. Step Event Test
    # ---------------------------------------------------------
    # Add steps with height 0.5 to 0.8
    aug_step = StepEvent(probability=1.0, gain_range=(0.5, 0.8), injection_prob=1.0)
    out_step = aug_step(signal.clone())

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------
    
    
    

    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    
    # Plot Gaussian Results
    for i in range(3):
        axs[0].plot(out_gauss[i, 0].numpy(), label=f"Sample {i}")
    axs[0].set_title("Gaussian Events (Transient Spikes)")
    axs[0].set_ylim(0, 1.5)
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    
    # Plot Step Results
    for i in range(3):
        axs[1].plot(out_step[i, 0].numpy(), label=f"Sample {i}")
    axs[1].set_title("Step Events (Baseline Shift)")
    axs[1].set_ylim(0, 1.5)
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    print("Event Augmentations test done.")