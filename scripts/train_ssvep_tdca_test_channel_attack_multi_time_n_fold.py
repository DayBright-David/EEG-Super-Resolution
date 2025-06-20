#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# 2024.8.15

'''
    本文件用于测试在Benchmark，Beta等数据上channel attack对于TDCA算法的准确率和ITR的影响。
    对5个时间窗（0.2-1s）进行交叉验证
    使用置零的方法进行channel attack

'''
import os
import sys
import numpy as np
import torch

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import utils
from utils import suggested_weights_filterbank, ITR, add_gaussian_white_noise, filterbank
from ssvep_tdca import SSVEP_TDCA, SSVEP_EAM_TDCA, SSVEP_CAM_TDCA, SSVEP_ROBUST_TDCA
from scipy.io import loadmat, savemat
from dataset import readSubjectData
import time
from timm.models import create_model
import argparse
import modeling_pretrain
from torch.cuda.amp import autocast

import warnings
warnings.filterwarnings('ignore')


target_noise_db = 0  # 对应噪声方差为1

isSuperResolution = True  # True则在测试阶段先mask，做超分辨再做TDCA，否则直接使用certain test data进行测试
isFinetune = False  # 如果不finetune，所有被试的所有fold都使用预训练模型参数




########### Benchmark ############
dataset_name = 'Benchmark'
sub_name_list = ['S'+str(idx) for idx in range(1, 36)]
# sub_name_list = ['S'+str(idx) for idx in [20]]
root_dir = '/home/daibo/classification_TDCA_TRF/dataset'
n_fold = 6
# 前九个通道为枕区九导
# 七导对比：['P5', 'P1', 'P2', 'P6', 'POZ', 'O1', 'O2']
# 十五导超分辨：['P5', 'P1', 'P2', 'P6', 'POZ', 'O1', 'O2', 'PZ', 'PO5', 'PO4', 'PO6', 'OZ', 'PO3', 'P3', 'P4']
#ssvep_selected_channels = ['P5', 'P1', 'P2', 'P6', 'POZ', 'O1', 'O2']
# ssvep_selected_channels = ['P5', 'P1', 'P2', 'P6', 'POZ', 'O1', 'O2', 'P3', 'P4', 'PO5', 'PO6', 'PO3', 'PO4', 'PZ', 'OZ']
# ssvep_selected_channels = ['P5', 'P6', 'P1', 'P2', 'PO3', 'PO4', 'P3', 'P4', 'PO5', 'PO6']
ssvep_selected_channels = ['P5', 'P6', 'P1', 'P2', 'PO3', 'PO4', 'P3', 'P4', 'PO5', 'PO6', 'O1', 'O2', 'PZ', 'OZ', 'POZ']
fs = 250
num_subbands = 5
harmonic_num = 5
delay_num = 0  # 数据增强时延迟的个数


########### Beta dataset ############
# dataset_name = 'Beta'
# sub_name_list = ['S'+str(idx) for idx in range(1, 71)]
# root_dir = 'F:/Xinyu Mou/classification_TDCA_TRF_8_15/dataset'
# ssvep_selected_channels = ['PZ', 'PO5', 'POZ', 'PO4', 'PO6', 'O1', 'OZ', 'PO3', 'O2', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7',
#      'P8', 'PO7', 'PO8', 'CB1', 'CB2']
# fs = 250
# n_fold = 4
# num_subbands = 5
# harmonic_num = 5
# delay_num = 0  # 数据增强时延迟的个数
# electrodes_adjacent_matrix = np.loadtxt(root_dir + '/' + dataset_name + '/Beta_9ch_normalized_adjacency_matrix.csv', delimiter=',')


CHANNELS_7 = ['P5', 'P1', 'P2', 'P6', 'POZ', 'O1', 'O2'] # 您的7通道定义
CHANNELS_15 = ['P5', 'P6', 'P1', 'P2', 'PO3', 'PO4', 'P3', 'P4', 'PO5', 'PO6', 'O1', 'O2', 'PZ', 'OZ', 'POZ'] # 您的15通道定义
ALL_AVAILABLE_CHANNELS = CHANNELS_15 # 或者数据集中所有可用的通道列表，确保15通道是其子集


def get_args():
    parser = argparse.ArgumentParser('LaBraM test script', add_help=False)

    # Model parameters
    parser.add_argument('--model', default='reconstruction_base_patch250_250', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_true', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=False)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=True)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float,
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")

    parser.add_argument('--input_size', default=250, type=int,
                        help='EEG input size for backbone')

    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')

    parser.add_argument('--finetune_checkpoint_path',
                        default='./finetune_checkpoints', type=str,
                        help='checkpoint path')

    parser.add_argument('--pretrain_checkpoint_path',
                        default='.checkpoints', type=str,
                        help='checkpoint path')

    parser.add_argument('--checkpoint_epoch',
                        default=199, type=int,
                        help='checkpoint epoch')

    parser.add_argument('--checkpoint_base_path', default='./checkpoints', type=str, help='Base path for user-specific checkpoint folders')
    parser.add_argument('--experiment_mode', default='7to15_reconstruct', type=str,
                        choices=['7channel_direct', '15channel_direct', '7to15_reconstruct'],
                        help='Experiment mode to run.')
    parser.add_argument('--target_channels_for_tdca', default='15', type=str, choices=['7', '15'],
                        help='Number of channels to use for TDCA classification (7 or 15).')

    return parser.parse_args()



def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_block_rate=None,
        use_shared_rel_pos_bias=args.rel_pos_bias,
        use_abs_pos_emb=args.abs_pos_emb,
        init_values=args.layer_scale_init_value
    )

    return model

def n_fold_evaluation(args, dataset_name, subject_name, clf_1, super_resolution_net, n_fold, train_winLEN, test_winLEN, num_subbands=5, harmonic_num=5, device='cuda:0'):
    n_fold_accs = []
    n_fold_itrs = []

    for fold_idx in range(n_fold):
        fold_acc, fold_itr = evaluation_on_attacked_data(args, dataset_name, subject_name, clf_1, super_resolution_net, fold_idx, train_winLEN, test_winLEN, num_subbands, harmonic_num, device)
        n_fold_accs.append(fold_acc)
        n_fold_itrs.append(fold_itr)

    return n_fold_accs, n_fold_itrs  # (6, )


def evaluation_on_attacked_data(args, dataset_name, subject_name, clf_1, super_resolution_net, fold_idx, train_winLEN, test_winLEN, num_subbands=5, harmonic_num=5, device='cuda:0'):
    
    current_data_load_channels = [] # Channels for initial data loading
    tdca_channels = [] # Channels for TDCA training and final testing
    perform_super_resolution = False
    apply_direct_noise_to_15ch = False # New flag: True if we load 15ch and noise 8 of them for direct TDCA

    # Determine channel configurations based on experiment mode
    if args.experiment_mode == '7channel_direct': # This mode is now 15 channels with 8 noisy
        current_data_load_channels = CHANNELS_15 # Load 15 channels
        tdca_channels = CHANNELS_15              # TDCA will use 15 channels
        apply_direct_noise_to_15ch = True
        perform_super_resolution = False
        if args.target_channels_for_tdca != '15':
             print("Warning: For modified '7channel_direct' (now 15ch with 8 noisy), target_channels_for_tdca is effectively 15. Overriding if necessary.")
             # args.target_channels_for_tdca = '15' # Ensure consistency if other parts of script use this arg directly for this mode
    elif args.experiment_mode == '15channel_direct': # Clean 15 channel direct
        current_data_load_channels = CHANNELS_15
        if args.target_channels_for_tdca == '7':
            tdca_channels = CHANNELS_7
        else: # '15'
            tdca_channels = CHANNELS_15
        perform_super_resolution = False
    elif args.experiment_mode == '7to15_reconstruct': # SR: 15ch (8 masked/noised) -> SR -> 15ch TDCA
        current_data_load_channels = CHANNELS_15 
        perform_super_resolution = True
        tdca_channels = CHANNELS_15 # SR reconstructs to 15, TDCA uses 15
        if args.target_channels_for_tdca != '15':
            print("Warning: For 7to15_reconstruct mode, target_channels_for_tdca should be 15.")
    else:
        raise ValueError(f"Unknown experiment_mode: {args.experiment_mode}")

    # --- Load Super-Resolution Model if needed (for the specific subject) ---
    # Model instance is passed, load its state_dict here if SR is performed
    if perform_super_resolution and super_resolution_net is not None:
        user_specific_checkpoint_folder = f"user_{subject_name.replace('S','')}"
        user_checkpoint_path = os.path.join(args.checkpoint_base_path, user_specific_checkpoint_folder)
        checkpoint_file = os.path.join(user_checkpoint_path, f'checkpoint-{args.checkpoint_epoch}.pth')
        
        if not os.path.exists(checkpoint_file):
            print(f"ERROR: Checkpoint file not found for user {subject_name} at {checkpoint_file}")
            return None, None # Return None to indicate failure for this fold
        
        print(f"Loading SR model weights for user {subject_name} from {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        checkpoint_model = checkpoint.get('model', checkpoint)
        # Add key cleaning logic if necessary (like in the old script)
        # Example:
        # state_dict_sr_net = super_resolution_net.state_dict()
        # for k in ['head.weight', 'head.bias']:
        #     if k in checkpoint_model and checkpoint_model[k].shape != state_dict_sr_net[k].shape:
        #         print(f"SR Net: Removing key {k} from pretrained checkpoint")
        #         del checkpoint_model[k]
        # all_keys = list(checkpoint_model.keys())
        # for key_chk in all_keys:
        #     if "relative_position_index" in key_chk: # Or other problematic keys
        #         checkpoint_model.pop(key_chk)
        utils.load_state_dict(super_resolution_net, checkpoint_model)
        # super_resolution_net.eval() # Already done in main

    # --- Load Data for TDCA Training ---
    # TDCA is trained on 'tdca_channels'. For "15ch_8noisy_direct", this is clean 15ch.
    print(f"Loading TRAINING data with {len(tdca_channels)} channels for TDCA: {tdca_channels}")
    train_dict, _ = readSubjectData(dataset_name, subject_name, train_winLEN=train_winLEN,
                                            test_winLEN=test_winLEN,
                                            data_divide_method='block', test_block_idx=fold_idx,
                                            filterbank_num=num_subbands, harmonic_number=harmonic_num,
                                            root_dir=root_dir, chnNames=tdca_channels) 
    if train_dict is None:
        print(f"ERROR: Failed to load training data for {subject_name}, fold {fold_idx}")
        return None, None
    X_train = train_dict['data']
    Y_train = train_dict['label']
    train_ref_sig = train_dict['ref_sig']
    clf_1.fit(X=X_train, Y=Y_train, train_ref_sig=train_ref_sig)

    # --- Prepare Test Data ---
    X_test_for_tdca_processed = []
    Y_test_labels = []
    test_ref_sig_final = None
    test_class_num = 0

    if apply_direct_noise_to_15ch:
        print(f"Loading FULL 15 channels for direct TDCA with 8 noisy channels: {current_data_load_channels}")
        # Load 15 ch data (single band for modification before filterbank)
        _, common_test_dict = readSubjectData(dataset_name, subject_name, train_winLEN=train_winLEN,
                                                   test_winLEN=test_winLEN,
                                                   data_divide_method='block', test_block_idx=fold_idx,
                                                   filterbank_num=1, harmonic_number=harmonic_num, 
                                                   root_dir=root_dir, chnNames=current_data_load_channels) # Load 15 ch
        if common_test_dict is None:
            print(f"ERROR: Failed to load test data for 15ch_8noisy_direct for {subject_name}, fold {fold_idx}")
            return None, None

        X_test_raw_15ch = common_test_dict['data'] 
        Y_test_labels = common_test_dict['label']
        test_ref_sig_final = common_test_dict['ref_sig']
        test_class_num = len(test_ref_sig_final) if test_ref_sig_final is not None else 0

        all_trial_input_stacked = np.stack(X_test_raw_15ch, axis=0).squeeze(axis=1) # (trial, 15, time)
        
        channels_to_noise_names = [ch for ch in CHANNELS_15 if ch not in CHANNELS_7]
        indices_to_noise = [CHANNELS_15.index(ch_name) for ch_name in channels_to_noise_names]
        
        print(f"Adding noise to {len(indices_to_noise)} channels (indices: {indices_to_noise}) for direct 15ch TDCA.")
        all_trial_input_noised = all_trial_input_stacked.copy()
        
        if indices_to_noise: # Check if there are channels to noise
            noise_for_channels = add_gaussian_white_noise(all_trial_input_noised[:, indices_to_noise, :], target_noise_db=0, mode='noisePower')
            all_trial_input_noised[:, indices_to_noise, :] = noise_for_channels
        
        X_test_for_tdca_temp = [trial_data for trial_data in all_trial_input_noised] 
        X_test_for_tdca_processed = [filterbank(trial, srate=fs, num_subbands=num_subbands) for trial in X_test_for_tdca_temp]

    elif perform_super_resolution: # Scenario: 7to15_reconstruct (simulated 7ch input from 15ch, then SR)
        print(f"Loading FULL 15 channels for SR input (will be masked/noised): {current_data_load_channels}")
        _, common_test_dict = readSubjectData(dataset_name, subject_name, train_winLEN=train_winLEN,
                                                       test_winLEN=test_winLEN,
                                                       data_divide_method='block', test_block_idx=fold_idx,
                                                       filterbank_num=1, harmonic_number=harmonic_num,
                                                       root_dir=root_dir, chnNames=current_data_load_channels) # Load 15 ch
        if common_test_dict is None:
            print(f"ERROR: Failed to load test data for SR for {subject_name}, fold {fold_idx}")
            return None, None

        X_test_sr_input_raw_15ch = common_test_dict['data']
        Y_test_labels = common_test_dict['label']
        test_ref_sig_final = common_test_dict['ref_sig']
        test_class_num = len(test_ref_sig_final) if test_ref_sig_final is not None else 0

        all_trial_sr_input_stacked = np.stack(X_test_sr_input_raw_15ch, axis=0).squeeze(axis=1) # (trial, 15, time)
        
        channels_to_zero_out_names = [ch for ch in CHANNELS_15 if ch not in CHANNELS_7]
        indices_to_zero = [CHANNELS_15.index(ch_name) for ch_name in channels_to_zero_out_names]
        
        print(f"Masking (zeroing out) channels at indices: {indices_to_zero} to simulate 7-channel input.")
        all_trial_sr_input_masked = all_trial_sr_input_stacked.copy()
        all_trial_sr_input_masked[:, indices_to_zero, :] = 0
        # Optional: Add noise as per old script
        noise = add_gaussian_white_noise(all_trial_sr_input_masked[:, indices_to_zero, :], target_noise_db=0, mode='noisePower')
        all_trial_sr_input_masked[:, indices_to_zero, :] = noise

        all_trial_sr_input_expanded = np.expand_dims(all_trial_sr_input_masked, axis=2) # (trial, 15, 1, time)
        all_trial_sr_input_torch = torch.from_numpy(all_trial_sr_input_expanded).float().to(device)
        
        all_trial_reconstructed_np = None
        if super_resolution_net is not None:
            with torch.no_grad():
                with autocast(enabled=(device.type == 'cuda')):
                    reconstructed_tensor = super_resolution_net(all_trial_sr_input_torch) # NO input_chans needed
                    if reconstructed_tensor.ndim == 4 and reconstructed_tensor.shape[2] == 1:
                        reconstructed_tensor = reconstructed_tensor.squeeze(2)
                    all_trial_reconstructed_np = reconstructed_tensor.cpu().detach().numpy()
        else:
             print(f"ERROR: Super-resolution net is None, cannot perform reconstruction for {subject_name}, fold {fold_idx}.")
             return None, None # Cannot proceed without the SR model
        
        if all_trial_reconstructed_np is None or all_trial_reconstructed_np.shape[1] != len(tdca_channels):
             actual_ch_out = all_trial_reconstructed_np.shape[1] if all_trial_reconstructed_np is not None else "None"
             print(f"ERROR: Reconstructed channels ({actual_ch_out}) != TDCA target channels ({len(tdca_channels)}). Or reconstruction failed.")
             return None, None

        X_test_for_tdca_temp = [trial_data for trial_data in all_trial_reconstructed_np] 
        X_test_for_tdca_processed = [filterbank(trial, srate=fs, num_subbands=num_subbands) for trial in X_test_for_tdca_temp]

    else: # Scenarios 1 & 2: 7channel_direct or 15channel_direct
        print(f"Loading TEST data with {len(current_data_load_channels)} channels directly for TDCA: {current_data_load_channels}")
        _, test_dict = readSubjectData(dataset_name, subject_name, train_winLEN=train_winLEN,
                                              test_winLEN=test_winLEN,
                                              data_divide_method='block', test_block_idx=fold_idx,
                                              filterbank_num=num_subbands, harmonic_number=harmonic_num,
                                              root_dir=root_dir, chnNames=current_data_load_channels) 
        if test_dict is None:
            print(f"ERROR: Failed to load test data for direct TDCA for {subject_name}, fold {fold_idx}")
            return None, None
            
        X_test_for_tdca_processed = test_dict['data']
        Y_test_labels = test_dict['label']
        test_ref_sig_final = test_dict['ref_sig']
        test_class_num = len(test_ref_sig_final) if test_ref_sig_final is not None else 0

    if not X_test_for_tdca_processed or Y_test_labels is None or test_ref_sig_final is None:
        print(f"ERROR: Missing data for TDCA scoring for {subject_name}, fold {fold_idx}.")
        return None, None

    acc, _ = clf_1.score(X_test_for_tdca_processed, Y_test_labels, test_ref_sig_final)
    itr = ITR(test_class_num, acc, test_winLEN) if test_class_num > 0 else 0.0
    
    return acc, itr

def main():
    base_args = get_args()
    
    args = base_args 
    
    experiment_description = ''
    if args.experiment_mode == '7channel_direct': # This mode is now 15 channel with 8 noisy
        experiment_description = '15ch_8noisy_direct'
        # It's better to handle tdca_channels logic within evaluation_on_attacked_data
        # Ensure args.target_channels_for_tdca is consistent if used elsewhere for this mode's setup
        # For example, if some other part of the script strictly uses args.target_channels_for_tdca
        # to define behavior for '7channel_direct' mode, it might need adjustment or this argument
        # should be explicitly set to '15' when calling the script for this modified mode.
        # The shell script will handle passing --target_channels_for_tdca "15" for this.
    elif args.experiment_mode == '15channel_direct':
        experiment_description = '15ch_direct'
    elif args.experiment_mode == '7to15_reconstruct':
        experiment_description = '7to15_recon'
    else:
        # Fallback or error if an unexpected mode is somehow passed (though choices restrict this)
        experiment_description = args.experiment_mode 

    print(f"\n\n########### RUNNING EXPERIMENT: {experiment_description} ###########")
    print(f"Experiment Mode: {args.experiment_mode}, TDCA Channels: {args.target_channels_for_tdca}")

    acc_tdca_all = np.zeros((len(sub_name_list), 1, n_fold))
    itr_tdca_all = np.zeros((len(sub_name_list), 1, n_fold))

    super_resolution_net = None
    device = torch.device(args.device)
    if args.experiment_mode == '7to15_reconstruct':
         super_resolution_net = get_model(args) 
         super_resolution_net.to(device)
         super_resolution_net.eval() 

    exp_time_start = time.time()
    print(f"Starting processing for all subjects in experiment '{experiment_description}' sequentially...")
    
    for sub_idx, subject_name in enumerate(sub_name_list):
        print(f'======================== subject: {subject_name} for {experiment_description} ========================')
        winLEN = 1.0 
        print(f'******** window length: {winLEN}s **********')
        time_start = time.time()
        
        weights_filterbank = suggested_weights_filterbank()
        tdca_model = SSVEP_TDCA(weights_filterbank=weights_filterbank, n_delay=delay_num)
        
        n_fold_accs, n_fold_itrs = n_fold_evaluation(
            args, 
            dataset_name, 
            subject_name, 
            tdca_model, 
            super_resolution_net, 
            n_fold, 
            winLEN, 
            winLEN, 
            num_subbands, 
            harmonic_num, 
            device
        )
        
        if n_fold_accs is not None:
            acc_tdca_all[sub_idx, 0] = n_fold_accs
        else:
            print(f"Warning: No accuracy results for subject {subject_name} in {experiment_description}. Filling with NaN.")
            acc_tdca_all[sub_idx, 0] = np.full(n_fold, np.nan)
        
        if n_fold_itrs is not None:
            itr_tdca_all[sub_idx, 0] = n_fold_itrs
        else:
            print(f"Warning: No ITR results for subject {subject_name} in {experiment_description}. Filling with NaN.")
            itr_tdca_all[sub_idx, 0] = np.full(n_fold, np.nan)

        time_end = time.time()
        avg_acc_str = f"{np.mean(n_fold_accs):.4f}" if n_fold_accs and len(n_fold_accs) > 0 and not np.all(np.isnan(n_fold_accs)) else "N/A"
        print(f'Accuracies for {subject_name} ({experiment_description}): {n_fold_accs}')
        print(f'Average of n_fold for {subject_name} ({experiment_description}): {avg_acc_str}')
        print(f'Time for {subject_name} ({experiment_description}): {time_end - time_start:.2f}s')
    
    exp_time_end = time.time()
    print(f"Finished all subjects for experiment '{experiment_description}'. Total time: {exp_time_end - exp_time_start:.2f}s")

    save_prefix = f'results_{dataset_name}_{experiment_description}'
    np.save(f'{save_prefix}_acc.npy', acc_tdca_all)
    np.save(f'{save_prefix}_itr.npy', itr_tdca_all)
    print(f"Results for {experiment_description} saved.")

if __name__ == "__main__":
    main()
