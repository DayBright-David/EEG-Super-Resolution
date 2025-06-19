import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
plt.rcParams.update({'font.size': 24}) # Set global font size

dataset_name = 'Benchmark' # Or 'Beta' if you used that
results_dir = '.' # Directory where your .npy files are saved
output_dir = './plots' # Directory to save the generated plots

# Experiment labels for plotting
experiment_labels = [
    '15 Ch (8 Noisy) Direct',
    '15 Ch Direct',
    '7 to 15 Ch Recon'
]

# Corresponding file prefixes
file_prefixes = [
    f'results_{dataset_name}_15ch_8noisy_direct',
    f'results_{dataset_name}_15ch_direct',
    f'results_{dataset_name}_7to15_recon'
]

# Metrics to plot
metrics = ['acc', 'itr']
metric_titles = {
    'acc': 'Accuracy',
    'itr': 'ITR (bits/min)'
}

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- Helper function to load and process data ---
def load_and_process_data_for_lineplot(prefix, metric):
    """
    Loads .npy file, returns per-subject means and per-subject SEMs across folds.
    Expected data shape: (num_subjects, 1, n_fold)
    Returns:
        per_subject_means (np.array): shape (num_subjects,)
        per_subject_sems (np.array): shape (num_subjects,)
        n_subjects (int)
        n_folds (int)
    """
    try:
        filepath = os.path.join(results_dir, f"{prefix}_{metric}.npy")
        data = np.load(filepath) # Expected (num_subjects, 1, n_fold)
        if data.ndim == 3 and data.shape[1] == 1:
            num_subjects = data.shape[0]
            n_folds = data.shape[2]
            
            per_subject_means = np.mean(data, axis=2).squeeze(axis=1) # Shape: (num_subjects,)
            per_subject_stds = np.std(data, axis=2).squeeze(axis=1)   # Shape: (num_subjects,)
            
            per_subject_sems = per_subject_stds / np.sqrt(n_folds) if n_folds > 0 else np.zeros_like(per_subject_stds)
            return per_subject_means, per_subject_sems, num_subjects, n_folds
        elif data.ndim == 2: # Handle case where data might be (num_subjects, n_fold)
            num_subjects = data.shape[0]
            n_folds = data.shape[1]
            per_subject_means = np.mean(data, axis=1)
            per_subject_stds = np.std(data, axis=1)
            per_subject_sems = per_subject_stds / np.sqrt(n_folds) if n_folds > 0 else np.zeros_like(per_subject_stds)
            return per_subject_means, per_subject_sems, num_subjects, n_folds
        else:
            print(f"Warning: Data in {filepath} has unexpected shape {data.shape}. Expected (num_subjects, 1, n_fold) or (num_subjects, n_fold).")
            return None, None, 0, 0
    except FileNotFoundError:
        print(f"Error: File not found - {filepath}")
        return None, None, 0, 0
    except Exception as e:
        print(f"Error loading or processing {filepath}: {e}")
        return None, None, 0, 0

# --- Plotting ---
for metric in metrics:
    plt.figure(figsize=(20, 10)) # Adjusted figure size for larger fonts
    
    max_num_subjects = 0
    plot_data_for_metric = [] # Store (means, sems, label) for each condition

    for i, prefix in enumerate(file_prefixes):
        means, sems, num_s, num_f = load_and_process_data_for_lineplot(prefix, metric)
        if means is not None and sems is not None:
            plot_data_for_metric.append({'means': means, 'sems': sems, 'label': experiment_labels[i], 'num_s': num_s})
            if num_s > max_num_subjects:
                max_num_subjects = num_s
        else:
            print(f"Skipping {experiment_labels[i]} for {metric_titles[metric]} due to data loading issues.")

    if not plot_data_for_metric:
        print(f"No data loaded for metric {metric_titles[metric]}. Skipping plots.")
        plt.close()
        continue
        
    if max_num_subjects == 0:
        print(f"No subjects found for metric {metric_titles[metric]}. Skipping plot.")
        plt.close()
        continue

    subject_indices = np.arange(max_num_subjects)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Specified colors
    markers = ['o', 's', '^']

    for idx, data_dict in enumerate(plot_data_for_metric):
        current_subject_indices = np.arange(data_dict['num_s'])
        if data_dict['num_s'] > 0:
            plt.errorbar(current_subject_indices, data_dict['means'], yerr=data_dict['sems'], 
                         label=data_dict['label'], marker=markers[idx % len(markers)], capsize=5, linestyle='-', 
                         color=colors[idx % len(colors)], linewidth=2.5, markersize=8)
        else:
            print(f"Skipping plot for {data_dict['label']} as it has no subject data.")

    plt.xlabel("Subject Index", fontsize=24)
    plt.ylabel(metric_titles[metric], fontsize=24)
    plt.title(f'{metric_titles[metric]} per Subject ({dataset_name} Dataset)', fontsize=28)
    
    if max_num_subjects > 0:
        plt.xticks(subject_indices, [f"S{i+1}" for i in subject_indices], fontsize=20) 
        if max_num_subjects > 15: 
            tick_frequency = max(1, max_num_subjects // 10) 
            plt.xticks(subject_indices[::tick_frequency], [f"S{i+1}" for i in subject_indices[::tick_frequency]], rotation=45, ha="right", fontsize=20)
        else:
            plt.xticks(fontsize=20) # Apply fontsize to all ticks if not rotating
    
    plt.yticks(fontsize=20) # Set y-axis tick font size

    plt.legend(fontsize=20)
    plt.grid(axis='y', linestyle='--', linewidth=0.7)
    plt.grid(axis='x', linestyle=':', alpha=0.6, linewidth=0.7)
    plt.tight_layout(pad=1.5) # Add some padding
    
    plot_filename_line = os.path.join(output_dir, f'{dataset_name}_{metric}_per_subject_lineplot_styled.pdf')
    plt.savefig(plot_filename_line, bbox_inches='tight')
    print(f"Saved line plot: {plot_filename_line}")
    plt.close()

print("\nVisualization script finished.") 