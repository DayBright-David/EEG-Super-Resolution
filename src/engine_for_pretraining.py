# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------

from cgitb import enable
import math
import sys
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from einops import rearrange
from contextlib import nullcontext
from utils import add_gaussian_white_noise, MetricLogger, SmoothedValue, get_input_chans, get_rank

def random_masking(x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        # mask = np.hstack([
        #     np.zeros(len_keep),
        #     np.ones(L - len_keep),
        # ])
        # np.random.shuffle(mask)

        return mask.to(torch.bool)


def train_one_epoch(model: torch.nn.Module,
                    data_loader_list: Iterable, optimizer: torch.optim.Optimizer, patch_size: int,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, args=None):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_fn = nn.MSELoss()

    step_loader = 0
    for data_loader in data_loader_list:
        if len(data_loader) == 0:
            continue

        for step, (batch) in enumerate(metric_logger.log_every(data_loader, print_freq * args.gradient_accumulation_steps, header)):
            # assign learning rate & weight decay for each step
            it = start_steps + step + step_loader  # global training iteration
            if lr_schedule_values is not None or wd_schedule_values is not None:
                for i, param_group in enumerate(optimizer.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                    if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                        param_group["weight_decay"] = wd_schedule_values[it]

            samples = batch[0]  # (batch, channel, time)
            samples = samples.float().to(device, non_blocking=True)
            # (batch, channel, patch_num, patch_size)
            samples = rearrange(samples, 'B N (A T) -> B N A T', T=patch_size)
            # (batch, seq_len=channel*patch_num)
            if epoch < 500:
                mask_num = np.random.randint(1, 3)
            elif epoch < 1000:
                mask_num = np.random.randint(1, 5)
            elif epoch < 1500:
                mask_num = np.random.randint(1, 7)
            else:
                mask_num = np.random.randint(1, 8)

            bool_masked_pos = random_masking(samples.flatten(1, 2), mask_ratio=mask_num/15).to(device, non_blocking=True)

            # mask channels with gaussian white noise
            masked_samples = samples.clone()
            masked_samples.flatten(1, 2)[bool_masked_pos] = 0
            noise = add_gaussian_white_noise(masked_samples.cpu().flatten(1, 2)[bool_masked_pos.cpu()], target_noise_db=0,
                                             mode='noisePower')
            masked_samples.flatten(1, 2)[bool_masked_pos] = noise.float().to(device)

            my_context = model.no_sync if args.distributed and (step + 1) % args.gradient_accumulation_steps != 0 else nullcontext
            with my_context():
                with torch.cuda.amp.autocast(): # enabled=False
                    reconstruction = model(masked_samples)
                    # 只用被mask的patch计算loss
                    loss = loss_fn(reconstruction, samples.flatten(1, 2))

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training at rank {get_rank()}", force=True)
                
                sys.exit(1)

            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= args.gradient_accumulation_steps
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order, update_grad=(step + 1) % args.gradient_accumulation_steps == 0)
            loss_scale_value = loss_scaler.state_dict()["scale"]
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.zero_grad()

            torch.cuda.synchronize()
            

            metric_logger.update(loss=loss_value)
            metric_logger.update(loss_scale=loss_scale_value)
            min_lr = 10.
            max_lr = 0.
            for group in optimizer.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])

            metric_logger.update(lr=max_lr)
            metric_logger.update(min_lr=min_lr)
            weight_decay_value = None
            for group in optimizer.param_groups:
                if group["weight_decay"] > 0:
                    weight_decay_value = group["weight_decay"]
            metric_logger.update(weight_decay=weight_decay_value)
            metric_logger.update(grad_norm=grad_norm)

            if log_writer is not None:
                log_writer.update(loss=loss_value, head="loss")
                log_writer.update(loss_scale=loss_scale_value, head="opt")
                log_writer.update(lr=max_lr, head="opt")
                log_writer.update(min_lr=min_lr, head="opt")
                log_writer.update(weight_decay=weight_decay_value, head="opt")
                log_writer.update(grad_norm=grad_norm, head="opt")

                log_writer.set_step()

            if lr_scheduler is not None:
                lr_scheduler.step_update(start_steps + step + step_loader)
        step_loader += step
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def test_one_epoch(model: torch.nn.Module,
                    data_loader_list: Iterable, patch_size: int,
                    device: torch.device, epoch: int,
                    args=None):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_fn = nn.MSELoss()

    reconstructions_for_all_datasets = dict()

    for data_loader in data_loader_list:
        if len(data_loader) == 0:
            continue

        reconstruction_list_dataset = dict()
        for step, (batch) in enumerate(
                metric_logger.log_every(data_loader, print_freq * args.gradient_accumulation_steps, header)):

            reconstruction_dict_batch = dict()
            samples = batch[0]  # (batch, channel, time)
            samples = samples.float().to(device, non_blocking=True)
            # (batch, channel, patch_num, patch_size)
            samples = rearrange(samples, 'B N (A T) -> B N A T', T=patch_size)
            # (batch, seq_len=channel*patch_num)
            mask_num = np.random.randint(7, 8)
            bool_masked_pos = random_masking(samples.flatten(1, 2), mask_ratio=mask_num/15).to(device, non_blocking=True)

            # mask channels with gaussian white noise
            masked_samples = samples.clone()
            masked_samples.flatten(1, 2)[bool_masked_pos] = 0
            noise = add_gaussian_white_noise(masked_samples.cpu().flatten(1, 2)[bool_masked_pos.cpu()],
                                             target_noise_db=0,
                                             mode='noisePower')
            masked_samples.flatten(1, 2)[bool_masked_pos] = noise.float().to(device)

            reconstruction_dict_batch['groundtruth'] = samples.cpu().detach().numpy()
            reconstruction_dict_batch['masked'] = masked_samples.cpu().detach().numpy()
            reconstruction_dict_batch['bool_mask'] = bool_masked_pos.cpu()



            with torch.cuda.amp.autocast():  # enabled=False
                reconstruction = model(masked_samples)
                # 只用被mask的patch计算loss
                loss = loss_fn(reconstruction, samples.flatten(1, 2))
                reconstruction_dict_batch['reconstruction'] = reconstruction.cpu().detach().numpy()

            loss_value = loss.item()
            metric_logger.update(loss=loss_value)

            reconstruction_list_dataset[str(step)] = reconstruction_dict_batch

        reconstructions_for_all_datasets['Benchmark'] = reconstruction_list_dataset



    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return reconstructions_for_all_datasets

