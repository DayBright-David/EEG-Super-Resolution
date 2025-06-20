#!/usr/bin/env python
"""
Quick Start Example for EEG Super-Resolution

This script demonstrates how to:
1. Load a pretrained model
2. Perform EEG super-resolution on sample data
3. Evaluate the reconstruction quality
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from modeling_pretrain import reconstruction_base_patch250_250
from utils import get_model

def load_pretrained_model(checkpoint_path, device='cuda'):
    """Load a pretrained EEG super-resolution model"""
    print(f"Loading model from {checkpoint_path}")
    
    # Create model with correct parameters
    # The TemporalConv default out_chans=8, so embed_dim should be 250*8=2000
    # But reconstruction_base_patch250_250 uses embed_dim=250, so we need out_chans=1
    model = reconstruction_base_patch250_250(pretrained=False, out_chans=1)
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        print("Model loaded successfully!")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Using randomly initialized model for demonstration")
    
    model = model.to(device)
    model.eval()
    return model

def demonstrate_super_resolution(model, input_data, patch_size=250, device='cuda'):
    """Demonstrate EEG super-resolution on input data"""
    print("Performing EEG super-resolution...")
    
    # Convert to tensor if needed
    if isinstance(input_data, np.ndarray):
        input_data = torch.from_numpy(input_data).float()
    
    input_data = input_data.to(device)
    
    # Add batch dimension if needed
    if len(input_data.shape) == 2:
        input_data = input_data.unsqueeze(0)
    
    # Ensure data shape is correct: (batch, channel, time)
    batch_size, n_channels, n_timepoints = input_data.shape
    
    # Reshape to patches as expected by the model: (batch, channel, patch_num, patch_size)
    input_data = rearrange(input_data, 'B N (A T) -> B N A T', T=patch_size)
    
    with torch.no_grad():
        # Perform super-resolution
        reconstructed = model(input_data)
        
        # Reshape back to original format
        reconstructed = reconstructed.view(batch_size, n_channels, -1)
        
    return reconstructed.cpu().numpy()

def visualize_results(original, reconstructed, channels_to_show=5, sampling_rate=250):
    """Visualize original vs reconstructed EEG signals"""
    print("Visualizing results...")
    
    # Handle different shapes
    if len(original.shape) == 3:
        original = original[0]  # Take first sample from batch
    if len(reconstructed.shape) == 3:
        reconstructed = reconstructed[0]  # Take first sample from batch
        
    fig, axes = plt.subplots(channels_to_show, 1, figsize=(12, 8))
    
    for i in range(min(channels_to_show, original.shape[0])):
        if channels_to_show == 1:
            ax = axes
        else:
            ax = axes[i]
            
        # Plot original and reconstructed signals
        time_axis = np.arange(original.shape[-1]) / sampling_rate
        
        if i < reconstructed.shape[0] and reconstructed.shape[1] >= original.shape[1]:
            ax.plot(time_axis, original[i, :], 'b-', label='Original', alpha=0.7)
            ax.plot(time_axis, reconstructed[i, :original.shape[1]], 'r--', label='Reconstructed', alpha=0.7)
            ax.set_title(f'Channel {i+1}')
            ax.set_ylabel('Amplitude (μV)')
            ax.legend()
        else:
            ax.plot(time_axis, original[i, :], 'b-', label='Original', alpha=0.7)
            ax.set_title(f'Channel {i+1} (Input only)')
            ax.set_ylabel('Amplitude (μV)')
            ax.legend()
    
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig('eeg_super_resolution_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Results saved as 'eeg_super_resolution_demo.png'")

def main():
    """Main demonstration function"""
    print("EEG Super-Resolution Quick Start Demo")
    print("=" * 40)
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Model parameters
    patch_size = 250  # Must match model's patch_size
    
    # Example checkpoint path (update this to your actual checkpoint)
    checkpoint_path = "./checkpoints/labram_base/checkpoint-1999.pth"
    
    # Load model
    model = load_pretrained_model(checkpoint_path, device)
    
    # Generate or load sample data
    print("\nGenerating sample EEG data...")
    # Based on engine_for_pretraining.py, the model expects:
    # - Total sequence length of 15 patches after flattening (channel * patch_num_per_channel = 15)
    # - This can be achieved with 15 channels × 1 patch each, or other combinations
    # - Common EEG setups use 9-15 channels, so let's use 15 channels with 1 patch each
    
    batch_size = 1
    n_channels = 15  # 15 channels to match the model's position embedding
    n_patches_per_channel = 1  # 1 patch per channel to get total 15 patches
    n_timepoints_per_patch = patch_size  # 250 time points per patch
    n_timepoints_total = n_patches_per_channel * n_timepoints_per_patch  # Total time points per channel
    
    # Generate sample data: (batch, channels, total_time_points)
    sample_data = np.random.randn(batch_size, n_channels, n_timepoints_total) * 50  # Scale to μV range
    
    # Add some realistic EEG characteristics
    for ch in range(n_channels):
        # Add some alpha rhythm (8-12 Hz) with different frequencies per channel
        t = np.arange(n_timepoints_total) / 250  # time axis in seconds
        alpha_freq = 8 + (ch % 5) * 0.8  # Frequencies from 8-11.2 Hz
        sample_data[0, ch, :] += 20 * np.sin(2 * np.pi * alpha_freq * t)
        
        # Add some beta rhythm (13-30 Hz)
        beta_freq = 15 + (ch % 8) * 2  # Frequencies from 15-29 Hz
        sample_data[0, ch, :] += 10 * np.sin(2 * np.pi * beta_freq * t)
        
        # Add some noise
        sample_data[0, ch, :] += np.random.randn(n_timepoints_total) * 5
    
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Data will be reshaped to: ({batch_size}, {n_channels}, {n_patches_per_channel}, {patch_size})")
    print(f"After flattening patches: sequence length = {n_channels * n_patches_per_channel}")
    
    # Perform super-resolution
    reconstructed = demonstrate_super_resolution(model, sample_data, patch_size, device)
    print(f"Reconstructed data shape: {reconstructed.shape}")
    
    # Calculate reconstruction quality metrics
    if reconstructed.shape == sample_data.shape:
        mse = np.mean((sample_data - reconstructed) ** 2)
        correlation = np.corrcoef(sample_data.flatten(), reconstructed.flatten())[0, 1]
        print(f"\nReconstruction Quality:")
        print(f"MSE: {mse:.4f}")
        print(f"Correlation: {correlation:.4f}")
    else:
        print(f"\nNote: Output shape {reconstructed.shape} differs from input shape {sample_data.shape}")
        print("This is expected for super-resolution models that may change the temporal dimension")
    
    # Visualize results (show first 9 channels for better visualization)
    channels_to_show = min(9, n_channels)
    visualize_results(sample_data, reconstructed, channels_to_show)
    
    print("\nDemo completed successfully!")
    print("Next steps:")
    print("1. Replace sample data with real EEG data using dataset.py")
    print("2. Use a trained checkpoint for better results") 
    print("3. Fine-tune the model on your specific dataset")
    print("4. The model expects exactly 15 total patches (channels × patches_per_channel = 15)")
    print("5. For different channel setups, adjust patches_per_channel accordingly:")
    print("   - 15 channels × 1 patch = 15 total patches ✓")
    print("   - 9 channels × 1.67 patches ≈ not integer (need padding or different approach)")
    print("   - 5 channels × 3 patches = 15 total patches ✓")

if __name__ == "__main__":
    main() 