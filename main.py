# -*- coding: utf-8 -*-
"""
Image and Signal Augmentation Pipeline.

This script demonstrates temporal (1D) and spatial (2D) data augmentation
using PyTorch and custom configuration files.

Author: Diyar Altinses, M.Sc.
Date:   2022-02-16 (Refactored: 2026-01-06)
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import yaml
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import PILToTensor

from utils.config_plots import configure_plt

configure_plt()

# Assuming 'augmentation.py' is a local module in the same directory
try:
    import augmentation as transform
except ImportError:
    logging.error("Module 'failure injection module' not found. Please ensure augmentation.py is in the working directory.")
    sys.exit(1)

# --- Configuration Constants ---
BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"
RESOURCE_DIR = BASE_DIR / "resources"
RESULTS_DIR = BASE_DIR / "results"

# Logging Setup
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Loads a YAML configuration file safely.
    
    Args:
        config_path (Path): Path to the .yaml file.
        
    Returns:
        Dict: Parsed configuration dictionary.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r") as f:
        try:
            # safe_load is generally preferred over FullLoader for security
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            logging.error(f"Error parsing YAML file: {e}")
            raise


def run_temporal_test(config_name: str = "config.yaml") -> None:
    """
    Executes the temporal (1D signal) augmentation test with improved visualization.
    Uses Heatmaps for batch overview and Line plots for detail.
    """
    logging.info("Starting temporal (1D) injection test...")
    
    config_path = CONFIG_DIR / config_name
    aug_dict = load_config(config_path)
    
    # Initialize Augmentation
    augmentation_pipeline = transform.Compose(aug_dict)

    # --- Generate Synthetic Data ---
    # Batch=16, Channels=1, Length=100
    # We create sine waves with different phases so they look like distinct signals
    batch_size = 16
    length = 100
    t = torch.linspace(0, 4 * 3.14159, length)
    
    # Create base signals (sine waves)
    base_signal = torch.sin(t).repeat(batch_size, 1, 1) 
    
    # Add an offset to each batch item so they have distinct mean values (like different sensors)
    offsets = torch.arange(batch_size).float().view(batch_size, 1, 1)
    data = base_signal + offsets

    # Apply Augmentation
    augmented_data = augmentation_pipeline(data)

    # --- Improved Visualization ---
    # We use a GridSpec layout: Top row for Heatmaps, Bottom row for specific line details
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

    # 1. MACRO VIEW: Heatmaps
    # This shows the entire batch at once. "Failure" (zeros) will appear as dark voids.
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Squeeze to (Batch, Time) for 2D plotting
    orig_map = data[:, 0, :].numpy()
    aug_map = augmented_data[:, 0, :].numpy()

    # Determine common color scale for fair comparison
    vmin, vmax = orig_map.min(), orig_map.max()

    ax1.imshow(orig_map, aspect='auto', cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax)
    ax1.set_title("Original Batch")
    ax1.set_ylabel("Batch Sample Index")
    ax1.set_xlabel("Time Step")

    im2 = ax2.imshow(aug_map, aspect='auto', cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax)
    ax2.set_title("Corrupted Batch")
    ax2.set_xlabel("Time Step")
    
    # Add colorbar
    fig.colorbar(im2, ax=[ax1, ax2], fraction=0.046, pad=0.04, label="Signal Amplitude")

    # 2. MICRO VIEW: Single Instance Line Plot
    # Pick the middle sample (e.g., index 8) to see exactly what the noise/failure looks like
    sample_idx = 8
    ax3 = fig.add_subplot(gs[1, :]) # Span full width
    
    ax3.plot(data[sample_idx, 0, :].numpy(), label="Original Signal", color='grey', alpha=0.5, linestyle='--')
    ax3.plot(augmented_data[sample_idx, 0, :].numpy(), label="Augmented Signal (With Injection)", color='#d62728', linewidth=1.5)
    
    ax3.set_title(f"Single Sample (Index {sample_idx})")
    ax3.set_xlabel("Time Step")
    ax3.set_ylabel("Amplitude")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.suptitle(f"Temporal Injection Analysis\n(Config: {config_name})", fontsize=14)
    plt.show()

def run_spatial_test(config_name: str = "spatial.yaml", image_name: str = "00000_frame.png") -> None:
    """
    Executes the spatial (2D image) failure injection test and saves results.
    """
    logging.info("Starting spatial (2D) injection test...")

    config_path = CONFIG_DIR / config_name
    image_path = RESOURCE_DIR / image_name

    if not image_path.exists():
        logging.error(f"Image not found at {image_path}")
        return

    # Load Config and Image
    aug_dict = load_config(config_path)
    augmentation_pipeline = transform.Compose(aug_dict)
    
    # Prepare Output Directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Image Preprocessing
    try:
        pil_image = Image.open(image_path)
        tensor_transform = PILToTensor()
        # Ensure we only take first 3 channels (RGB), ignoring Alpha if present
        image_tensor = tensor_transform(pil_image)[:3] 
        
        # Create a batch of 16 identical images
        batch = image_tensor.repeat(16, 1, 1, 1)
    except Exception as e:
        logging.error(f"Failed to process image: {e}")
        return

    logging.info(f"Processing batch of shape: {batch.shape}")

    # Process and Save
    for i in range(len(batch)):
        transformed_image = augmentation_pipeline(batch[i])
        
        fig = plt.figure(figsize=(3, 3))
        # Permute from (C, H, W) to (H, W, C) for Matplotlib
        plt.imshow(transformed_image.permute(1, 2, 0).numpy())
        plt.axis('off')
        
        output_file = RESULTS_DIR / f"aug_{i:02d}.png"
        fig.savefig(output_file, dpi=300, pad_inches=0.01, bbox_inches='tight')
        
    logging.info(f"Saved {len(batch)} images to {RESULTS_DIR}")


def main():
    """Main entry point of the application."""
    
    # torch.manual_seed(42) # Ensure reproducibility 
    try:
        run_temporal_test()
        run_spatial_test()
    except Exception as e:
        logging.critical(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()