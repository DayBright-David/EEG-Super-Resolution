# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------

import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os

from pathlib import Path

from timm.models import create_model
from optim_factory import create_optimizer

from engine_for_pretraining import train_one_epoch, test_one_epoch
import utils
from utils import NativeScalerWithGradNormCount as NativeScaler
from dataset import build_Benchmark_dataset_for_pretraining
import modeling_pretrain


def get_args():
    parser = argparse.ArgumentParser('LaBraM test script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)


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

    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)


    # checkpoint
    parser.add_argument('--start_epoch', default=4, type=int)
    parser.add_argument('--epochs', default=2000, type=int)
    parser.add_argument('--epoch_interval', default=5, type=int)
    parser.add_argument('--checkpoint_path', default='/home/daibo/self_supervised_reconstruction/checkpoints/labram_base', type=str,
                        help='checkpoint path')

    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)


    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    

    # dataset parameters
    parser.add_argument('--dataset_root_dir', default='../dataset/Benchmark', type=str,
                        help='Root path of the dataset')
    parser.add_argument('--srate', default=250, type=int,
                        help='sampling rate of the dataset')


    # reconstruction save parameters
    parser.add_argument('--reconstruction_save_path', default='./reconstruction', type=str,
                        help='Path to save all reconstructions')

    # target user id
    parser.add_argument('--user_id', default=1, type=int,
                        help='target user id')


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


def main(args):

    if not os.path.isdir(args.reconstruction_save_path):
        os.makedirs(args.reconstruction_save_path)
        print('create dirs: ', args.reconstruction_save_path)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    model = get_model(args)
    patch_size = model.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (1, args.input_size // patch_size)
    args.patch_size = patch_size


    dataset_train, dataset_test = build_Benchmark_dataset_for_pretraining(root_dir=args.dataset_root_dir, winLEN=args.input_size/args.srate, remove_idx=args.user_id)


    dataset_test_list = [dataset_test]

    data_loader_test_list = []
    for dataset in dataset_test_list:
        data_loader_test = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
        data_loader_test_list.append(data_loader_test)

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * args.gradient_accumulation_steps
    print("Batch size = %d" % total_batch_size)

    print(f"Start testing")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs+1, args.epoch_interval):


        checkpoint = torch.load(args.checkpoint_path + '/checkpoint-' + str(epoch) + '.pth', map_location='cpu')

        checkpoint_model = checkpoint['model']
        if checkpoint_model is None:
            checkpoint_model = checkpoint

        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        for key in all_keys:
            if "relative_position_index" in key:
                checkpoint_model.pop(key)

        utils.load_state_dict(model, checkpoint_model)

        reconstructions_for_all_datasets = test_one_epoch(
            model, data_loader_test_list,
            args.patch_size, device, epoch,
            args=args,
        )

        save_path = args.reconstruction_save_path + '/reconstruction_epoch_' + str(epoch) + '.npy'
        np.save(save_path, reconstructions_for_all_datasets)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Testing time {}'.format(total_time_str))


if __name__ == '__main__':
    opts = get_args()
    main(opts)
