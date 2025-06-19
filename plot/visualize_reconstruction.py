import torch
import numpy as np
import matplotlib.pyplot as plt
from src.plot_methods import plot_eeg
import os

# --- Configuration ---
# IMPORTANT: Ensure this file exists. Based on user_1.log, epoch 1999 might not exist for user_1 if --epochs was 1004.
file_path = 'reconstruct_results/user_1/reconstruction_epoch_1999.npy'
dataset_key = 'Benchmark'
visualize_batch_idx = 0
visualize_sample_idx = 5
visualize_ch_idx = 1 # Channel index to visualize (0-indexed)
srate = 250 # Sampling rate in Hz, adjust if necessary
output_dir = 'reconstruction_visualization/user_1'

# --- Load Data ---
try:
    reconstruction_for_all_datasets = np.load(file_path, allow_pickle=True).item()
except FileNotFoundError:
    print(f"ERROR: Data file not found at {file_path}")
    print("Please ensure the file exists or update the 'file_path' variable in the script.")
    exit()
except Exception as e:
    print(f"ERROR: Could not load or parse the .npy file: {e}")
    exit()

if dataset_key not in reconstruction_for_all_datasets:
    print(f"ERROR: Dataset key '{dataset_key}' not found in the loaded .npy file.")
    exit()
reconstruction_for_benchmark = reconstruction_for_all_datasets[dataset_key]

batch_idx_str = str(visualize_batch_idx)
if batch_idx_str not in reconstruction_for_benchmark:
    print(f"ERROR: Batch index '{batch_idx_str}' not found for dataset '{dataset_key}'.")
    exit()
batch_dict = reconstruction_for_benchmark[batch_idx_str]

# Extract relevant data
try:
    batch_groundtruth = batch_dict['groundtruth']
    # batch_masked = batch_dict['masked'] # Not used in this specific plot
    batch_reconstruction = batch_dict['reconstruction']
    # batch_bool_mask = batch_dict['bool_mask'] # Not used in this specific plot
except KeyError as e:
    print(f"ERROR: Missing key {e} in batch_dict.")
    exit()

# Select the specific sample
if not isinstance(batch_groundtruth, np.ndarray) or not isinstance(batch_reconstruction, np.ndarray):
    print(f"ERROR: 'groundtruth' or 'reconstruction' is not a NumPy array.")
    exit()

if visualize_sample_idx >= batch_groundtruth.shape[0]:
    print(f"ERROR: visualize_sample_idx {visualize_sample_idx} is out of bounds for batch_groundtruth with {batch_groundtruth.shape[0]} samples.")
    exit()

visualize_sample_groundtruth_all_ch = batch_groundtruth[visualize_sample_idx].squeeze()
visualize_sample_reconstruction_all_ch = batch_reconstruction[visualize_sample_idx].squeeze()

# Validate shapes and channel index
if visualize_sample_groundtruth_all_ch.ndim != 2 or visualize_sample_reconstruction_all_ch.ndim != 2:
    print(f"ERROR: Expected 2D arrays for sample data (num_channels, timepoints).")
    exit()
if visualize_ch_idx >= visualize_sample_groundtruth_all_ch.shape[0]:
    print(f"ERROR: visualize_ch_idx {visualize_ch_idx} is out of bounds for data with {visualize_sample_groundtruth_all_ch.shape[0]} channels.")
    exit()

visualize_sample_groundtruth = visualize_sample_groundtruth_all_ch[visualize_ch_idx]
visualize_sample_reconstruction = visualize_sample_reconstruction_all_ch[visualize_ch_idx]
num_timepoints = visualize_sample_groundtruth.shape[0]
time_axis = np.arange(num_timepoints) / srate

# --- Plotting --- 
plt.style.use('seaborn-v0_8-whitegrid') # Using a seaborn style for better aesthetics
plt.figure(figsize=(14, 7)) # Slightly larger figure

plt.plot(time_axis, visualize_sample_groundtruth, label='Ground Truth', color='dodgerblue', linewidth=2)
plt.plot(time_axis, visualize_sample_reconstruction, label='Reconstruction', color='orangered', linestyle='--', linewidth=2)

plt.title(f'EEG Signal: Ground Truth vs. Reconstruction\nChannel {visualize_ch_idx + 1} (Epoch 1999, Batch {visualize_batch_idx}, Sample {visualize_sample_idx})', fontsize=24)
plt.xlabel('Time (s)', fontsize=24)
plt.ylabel('Amplitude', fontsize=24)
plt.legend(fontsize=24)
plt.grid(True, linestyle=':', alpha=0.7)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

# --- Save Figure ---
os.makedirs(output_dir, exist_ok=True)
output_filename = f'channel_{visualize_ch_idx + 1}_comparison_epoch_1999_batch_{visualize_batch_idx}_sample_{visualize_sample_idx}.pdf'
output_path = os.path.join(output_dir, output_filename)

try:
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Plot saved to {output_path}")
except Exception as e:
    print(f"Error saving plot: {e}")

# plt.show() # Uncomment to display plot interactively

# --- Print numerical data (optional) ---
# print("Ground Truth Data:")
# print(visualize_sample_groundtruth)
# print("\nReconstruction Data:")
# print(visualize_sample_reconstruction)
# print("\nDifference (Ground Truth - Reconstruction):")
# print(visualize_sample_groundtruth - visualize_sample_reconstruction)
