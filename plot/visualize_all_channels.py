import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
# Path to the reconstruction results file
# IMPORTANT: Ensure this file exists. Based on user_1.log, epoch 1999 might not exist for user_1 if --epochs was 1004.
# Modify if your file is different.
file_path = 'reconstruct_results/user_1/reconstruction_epoch_1999.npy' 
dataset_key = 'Benchmark' # Key for the dataset in the .npy file
visualize_batch_idx = 0   # Batch index to visualize
visualize_sample_idx = 5  # Sample index within the batch to visualize
output_dir = 'reconstruction_visualization/user_1'
# Ensure the output filename is unique if you change epochs or samples
output_filename = 'all_channels_comparison_epoch_1999_batch_{}_sample_{}.pdf'.format(visualize_batch_idx, visualize_sample_idx)
srate = 250 # Sampling rate for time axis calculation (Hz) - adjust if different for your data

# --- Load Data ---
try:
    reconstruction_data_loaded = np.load(file_path, allow_pickle=True).item()
except FileNotFoundError:
    print(f"ERROR: Data file not found at {file_path}")
    print("Please ensure the file exists or update the 'file_path' variable in the script.")
    exit()
except Exception as e:
    print(f"ERROR: Could not load or parse the .npy file: {e}")
    exit()

if dataset_key not in reconstruction_data_loaded:
    print(f"ERROR: Dataset key '{dataset_key}' not found in the loaded .npy file.")
    print(f"Available keys: {list(reconstruction_data_loaded.keys())}")
    exit()
reconstruction_for_dataset = reconstruction_data_loaded[dataset_key]

batch_idx_str = str(visualize_batch_idx)
if batch_idx_str not in reconstruction_for_dataset:
    print(f"ERROR: Batch index '{batch_idx_str}' not found for dataset '{dataset_key}'.")
    print(f"Available batch keys: {list(reconstruction_for_dataset.keys())}")
    exit()
batch_dict = reconstruction_for_dataset[batch_idx_str]

# Extract groundtruth and reconstruction
try:
    batch_groundtruth = batch_dict['groundtruth']
    batch_reconstruction = batch_dict['reconstruction']
except KeyError as e:
    print(f"ERROR: Missing key {e} in batch_dict. Ensure 'groundtruth' and 'reconstruction' keys exist.")
    print(f"Available keys in batch_dict: {list(batch_dict.keys())}")
    exit()

# Select the specific sample
if not isinstance(batch_groundtruth, np.ndarray) or not isinstance(batch_reconstruction, np.ndarray):
    print(f"ERROR: 'groundtruth' or 'reconstruction' is not a NumPy array.")
    exit()
    
if visualize_sample_idx >= batch_groundtruth.shape[0]:
    print(f"ERROR: visualize_sample_idx {visualize_sample_idx} is out of bounds for batch_groundtruth with {batch_groundtruth.shape[0]} samples.")
    exit()
    
visualize_sample_groundtruth = batch_groundtruth[visualize_sample_idx].squeeze() 
visualize_sample_reconstruction = batch_reconstruction[visualize_sample_idx].squeeze()

# Validate shapes after squeeze - should be (num_channels, timepoints)
if visualize_sample_groundtruth.ndim != 2 or visualize_sample_reconstruction.ndim != 2:
    print(f"ERROR: Expected 2D arrays for sample groundtruth/reconstruction (num_channels, timepoints) after squeeze.")
    print(f"Groundtruth shape: {visualize_sample_groundtruth.shape}, Reconstruction shape: {visualize_sample_reconstruction.shape}")
    exit()

if visualize_sample_groundtruth.shape != visualize_sample_reconstruction.shape:
    print(f"ERROR: Groundtruth and reconstruction shapes do not match!")
    print(f"Groundtruth shape: {visualize_sample_groundtruth.shape}, Reconstruction shape: {visualize_sample_reconstruction.shape}")
    exit()

num_channels, num_timepoints = visualize_sample_groundtruth.shape
time_axis = np.arange(num_timepoints) / srate

# --- Plotting ---
# Adjust figsize for better readability depending on the number of channels
# Increase vertical space per subplot if channel labels or titles overlap
fig_height_per_channel = 1.0 # Inches per channel
fig_width = 15
fig_height = max(6, num_channels * fig_height_per_channel) # Ensure a minimum height

fig, axes = plt.subplots(num_channels, 1, figsize=(fig_width, fig_height), sharex=True)

if num_channels == 1: # If only one channel, axes is not an array, make it a list
    axes = [axes]

fig.suptitle(f'Ground Truth vs. Reconstruction - Epoch 1999, Batch {visualize_batch_idx}, Sample {visualize_sample_idx}', fontsize=16, y=0.99)

for i in range(num_channels):
    ax = axes[i]
    ax.plot(time_axis, visualize_sample_groundtruth[i], label='Ground Truth', color='dodgerblue', linewidth=1.2)
    ax.plot(time_axis, visualize_sample_reconstruction[i], label='Reconstruction', color='orangered', linestyle='--', linewidth=1.2)
    ax.set_ylabel(f'Ch {i+1}', fontsize=10)
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    if i == 0: # Add legend to the first subplot only
        ax.legend(loc='upper right', fontsize=10)

axes[-1].set_xlabel('Time (s)', fontsize=12)
axes[-1].tick_params(axis='x', labelsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to make space for suptitle

# --- Save Figure ---
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, output_filename)
try:
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Plot saved to {output_path}")
except Exception as e:
    print(f"Error saving plot: {e}")

# plt.show() # Uncomment to display the plot interactively
