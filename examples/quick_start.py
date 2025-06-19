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

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from modeling_pretrain import reconstruction_base_patch250_250
from utils import get_model
from dataset import readSubjectData

def load_pretrained_model(checkpoint_path, device='cuda'):
    """Load a pretrained EEG super-resolution model"""
    print(f"Loading model from {checkpoint_path}")
    
    # Create model
    model = reconstruction_base_patch250_250(pretrained=False)
    
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

def demonstrate_super_resolution(model, input_data, device='cuda'):
    """Demonstrate EEG super-resolution on input data"""
    print("Performing EEG super-resolution...")
    
    # Convert to tensor if needed
    if isinstance(input_data, np.ndarray):
        input_data = torch.from_numpy(input_data).float()
    
    input_data = input_data.to(device)
    
    # Add batch dimension if needed
    if len(input_data.shape) == 2:
        input_data = input_data.unsqueeze(0)
    
    with torch.no_grad():
        # Perform super-resolution
        reconstructed = model(input_data)
        
    return reconstructed.cpu().numpy()

def visualize_results(original, reconstructed, channels_to_show=5):
    """Visualize original vs reconstructed EEG signals"""
    print("Visualizing results...")
    
    fig, axes = plt.subplots(channels_to_show, 1, figsize=(12, 8))
    
    for i in range(min(channels_to_show, original.shape[1])):
        if channels_to_show == 1:
            ax = axes
        else:
            ax = axes[i]
            
        # Plot original and reconstructed signals
        time_axis = np.arange(original.shape[-1]) / 250  # Assuming 250 Hz sampling rate
        
        if i < reconstructed.shape[1]:
            ax.plot(time_axis, original[0, i, :], 'b-', label='Original', alpha=0.7)
            ax.plot(time_axis, reconstructed[0, i, :], 'r--', label='Reconstructed', alpha=0.7)
            ax.set_title(f'Channel {i+1}')
            ax.set_ylabel('Amplitude (μV)')
            ax.legend()
        else:
            ax.plot(time_axis, original[0, i, :], 'b-', label='Original', alpha=0.7)
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
    
    # Example checkpoint path (update this to your actual checkpoint)
    checkpoint_path = "./checkpoints/labram_base/checkpoint-1999.pth"
    
    # Load model
    model = load_pretrained_model(checkpoint_path, device)
    
    # Generate or load sample data
    print("\nGenerating sample EEG data...")
    # This is synthetic data for demonstration
    # In practice, you would load real EEG data using the dataset module
    batch_size, n_channels, n_timepoints = 1, 9, 1000  # 9 channels, 4 seconds at 250 Hz
    sample_data = np.random.randn(batch_size, n_channels, n_timepoints) * 50  # Scale to μV range
    
    # Add some realistic EEG characteristics
    for ch in range(n_channels):
        # Add some alpha rhythm (8-12 Hz)
        t = np.arange(n_timepoints) / 250
        alpha_freq = 10  # Hz
        sample_data[0, ch, :] += 20 * np.sin(2 * np.pi * alpha_freq * t)
    
    print(f"Sample data shape: {sample_data.shape}")
    
    # Perform super-resolution
    reconstructed = demonstrate_super_resolution(model, sample_data, device)
    print(f"Reconstructed data shape: {reconstructed.shape}")
    
    # Calculate reconstruction quality metrics
    if reconstructed.shape == sample_data.shape:
        mse = np.mean((sample_data - reconstructed) ** 2)
        correlation = np.corrcoef(sample_data.flatten(), reconstructed.flatten())[0, 1]
        print(f"\nReconstruction Quality:")
        print(f"MSE: {mse:.4f}")
        print(f"Correlation: {correlation:.4f}")
    
    # Visualize results
    visualize_results(sample_data, reconstructed)
    
    print("\nDemo completed successfully!")
    print("Next steps:")
    print("1. Replace sample data with real EEG data")
    print("2. Use a trained checkpoint for better results") 
    print("3. Fine-tune the model on your specific dataset")

if __name__ == "__main__":
    main() 