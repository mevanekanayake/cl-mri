import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
from torchvision.utils import make_grid

import platform
import json
from datetime import datetime
import pandas as pd
from collections import OrderedDict
import time
import numpy as np
from utils import metrics
import random
import logging
import sys
import os
from pathlib import Path
import copy

def set_seed(seed=9):
    torch.manual_seed(seed)  # torch
    random.seed(seed)  # python
    np.random.seed(seed)  # numpy
    torch.cuda.manual_seed(seed)  # torch.cuda


def set_cuda(deterministic=True, benchmark=False):  # set deterministic to True if the input size remains same
    cudnn.deterministic = deterministic
    cudnn.benchmark = benchmark


def fetch_paths(dataset):
    node = platform.node() if (platform.node()[:2] != "m3" and platform.node()[:2] != "dg") else "m3"
    path_file = os.path.join('utils', 'paths.json')
    f = json.load(open(path_file))
    data_path = Path(f[node][dataset])

    experiments_path = Path(f[node]["experiments"])
    folder_name = "Experiment_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    exp_path = Path(os.path.join(experiments_path, folder_name))
    os.makedirs(exp_path)

    return data_path, exp_path


def set_logger(exp_path):
    logger = logging.getLogger()
    filehandler = logging.FileHandler(os.path.join(exp_path, f'{exp_path.name}_logs.log'))
    streamhandler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(message)s')
    streamhandler.setFormatter(formatter)
    filehandler.setFormatter(formatter)
    logger.addHandler(streamhandler)
    logger.addHandler(filehandler)
    logger.setLevel(logging.INFO)

    return logger


def set_device(model, args):
    if args.dp:
        device_ids = list(range(torch.cuda.device_count())) if args.dp == "all" else [int(device_id) for device_id in args.dp.split(',')]
        model = nn.DataParallel(model, device_ids=device_ids)
        args.dv = f'cuda:{device_ids[0]}'
        model.to(args.dv)
    else:
        model.to(args.dv)
        device_ids = [0]

    return model, args, device_ids


class RunManager:
    epoch_start_time = float
    epoch_train_loss: torch.Tensor
    epoch_val_loss: torch.Tensor

    train_slice_count: int
    val_slice_count: int

    sequences: dict

    mse_vals: dict
    target_norms: dict
    ssim_vals: dict

    slice_stats: dict

    best_epoch: bool

    def __init__(self, experiments_path, ckpt, seq_types, plot_freq):

        self.experiments_path = experiments_path
        self.folder_name = experiments_path.name
        self.fig_path = os.path.join(self.experiments_path, f'{self.folder_name}_validation_images')
        os.mkdir(self.fig_path)
        self.epoch_count = ckpt['epoch'] if ckpt else 0
        self.best_model_state_dict = ckpt['best_model_state_dict'] if ckpt else None
        self.best_val_loss = ckpt['best_val_loss'] if ckpt else float('inf')
        self.seq_types = seq_types
        self.plot_freq = plot_freq
        self.summary = OrderedDict({"epoch_no": [],
                                    "epoch_duration": [],
                                    "train_loss": [],
                                    "val_loss": [],
                                    "val_NMSE": [],
                                    "val_PSNR": [],
                                    "val_SSIM": [],
                                    })
        for seq_type in self.seq_types:
            self.summary[f"{seq_type}_val_NMSE"] = []
            self.summary[f"{seq_type}_val_PSNR"] = []
            self.summary[f"{seq_type}_val_SSIM"] = []

    def begin_epoch(self):

        self.epoch_count += 1

        self.epoch_start_time = time.time()
        self.epoch_train_loss = torch.tensor(0.)
        self.epoch_val_loss = torch.tensor(0.)

        self.train_slice_count = 0
        self.val_slice_count = 0

        self.sequences = {}

        self.mse_vals = {}
        self.target_norms = {}
        self.maximum_vals = {}

        self.slice_stats = {}

        self.best_epoch = False

    def end_train_step(self, train_loss, size_of_train_batch):
        self.epoch_train_loss += train_loss * size_of_train_batch
        self.train_slice_count += size_of_train_batch

    def end_val_step(self, fnames, slice_nums, sequences, zfimages, outputs, targets, val_loss, max_values):

        size_of_val_batch = targets.shape[0]
        self.epoch_val_loss += val_loss * size_of_val_batch
        self.val_slice_count += size_of_val_batch

        for fname, slice_num, sequence, zfimage, output, target, max_value in zip(fnames, slice_nums, sequences, zfimages, outputs, targets, max_values):

            slice_num = slice_num.item()

            if fname not in self.sequences.keys():
                self.sequences[fname] = sequence
                self.mse_vals[fname] = {}
                self.target_norms[fname] = {}
                self.slice_stats[fname] = {}
                self.slice_stats[fname]["nmse"] = {}
                self.slice_stats[fname]["psnr"] = {}
                self.slice_stats[fname]["ssim"] = {}
                self.maximum_vals[fname] = {}

            self.mse_vals[fname][slice_num] = torch.mean((target - output) ** 2)
            self.target_norms[fname][slice_num] = torch.mean((target - torch.zeros_like(target)) ** 2)
            self.maximum_vals[fname][slice_num] = max_value
            
            # SAVING SLICE-WISE STATISTICS
            self.slice_stats[fname]["nmse"][slice_num] = (self.mse_vals[fname][slice_num] / self.target_norms[fname][slice_num]).item()
            self.slice_stats[fname]["psnr"][slice_num] = (10 * torch.log10(max_value** 2 / self.mse_vals[fname][slice_num])).item()
            self.slice_stats[fname]["ssim"][slice_num] = (metrics.ssim(target, output, max_value)).item()

    def end_epoch(self, model, optimizer, logger):

        epoch_duration = time.time() - self.epoch_start_time
        volume_stats = {}

        for fname, sequence in self.sequences.items():
            v_mse_val = torch.mean(torch.tensor(list(self.mse_vals[fname].values())))
            v_target_norm = torch.mean(torch.tensor(list(self.target_norms[fname].values())))
            v_max_val = torch.mean(torch.tensor(list(self.maximum_vals[fname].values())))
            # SAVING VOLUME-WISE STATISTICS
            volume_stats[fname] = {}
            volume_stats[fname]["nmse"] = v_mse_val / v_target_norm
            volume_stats[fname]["psnr"] = 10 * torch.log10(v_max_val ** 2 / v_mse_val)
            volume_stats[fname]["ssim"] = torch.mean(torch.tensor(list(self.slice_stats[fname]["ssim"].values())))
            volume_stats[fname]["num_slices"] = len(self.mse_vals[fname])

        avg_epoch_train_loss = self.epoch_train_loss / self.train_slice_count
        avg_epoch_val_loss = self.epoch_val_loss / self.val_slice_count

        # SAVE SUMMARY
        self.summary["epoch_no"].append(self.epoch_count)
        self.summary["epoch_duration"].append(epoch_duration)
        self.summary["train_loss"].append(avg_epoch_train_loss.item())
        self.summary["val_loss"].append(avg_epoch_val_loss.item())
        self.summary["val_NMSE"].append(torch.mean(torch.tensor([volume_stats[fname]["nmse"] for fname in volume_stats.keys()])).item())
        self.summary["val_PSNR"].append(torch.mean(torch.tensor([volume_stats[fname]["psnr"] for fname in volume_stats.keys()])).item())
        self.summary["val_SSIM"].append(torch.mean(torch.tensor([volume_stats[fname]["ssim"] for fname in volume_stats.keys()])).item())
        for seq_type in self.seq_types:
            self.summary[f"{seq_type}_val_NMSE"].append(torch.mean(torch.tensor([volume_stats[fname]["nmse"] for fname in volume_stats.keys() if self.sequences[fname] == seq_type])).item())
            self.summary[f"{seq_type}_val_PSNR"].append(torch.mean(torch.tensor([volume_stats[fname]["psnr"] for fname in volume_stats.keys() if self.sequences[fname] == seq_type])).item())
            self.summary[f"{seq_type}_val_SSIM"].append(torch.mean(torch.tensor([volume_stats[fname]["ssim"] for fname in volume_stats.keys() if self.sequences[fname] == seq_type])).item())

        pd.DataFrame.from_dict(self.summary, orient='columns').to_csv(Path(os.path.join(f'{self.experiments_path}', f'{self.folder_name}_summary.csv')), index=False)

        last_model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        last_optimizer_state_dict = optimizer.state_dict()

        # SAVE VOLUME/SLICE-WISE STATS and STATES (only if best epoch)
        if avg_epoch_val_loss < self.best_val_loss:
            self.best_val_loss = avg_epoch_val_loss
            self.best_model_state_dict = copy.deepcopy(last_model_state_dict)

            volume_slice_stats = OrderedDict({})

            fnames = volume_stats.keys()
            volume_slice_stats["fname"] = fnames
            volume_slice_stats["sequence"] = [self.sequences[fname] for fname in fnames]
            volume_slice_stats["num_slices"] = [volume_stats[fname]["num_slices"] for fname in fnames]
            volume_slice_stats["NMSE"] = [volume_stats[fname]["nmse"].item() for fname in fnames]
            volume_slice_stats["PSNR"] = [volume_stats[fname]["psnr"].item() for fname in fnames]
            volume_slice_stats["SSIM"] = [volume_stats[fname]["ssim"].item() for fname in fnames]
            for slice_idx in range(max(volume_slice_stats["num_slices"])):
                volume_slice_stats[f"NMSE_S{str(slice_idx + 1).zfill(2)}"] = [self.slice_stats[fname]["nmse"].get(slice_idx, '') for fname in fnames]
                volume_slice_stats[f"SSIM_S{str(slice_idx + 1).zfill(2)}"] = [self.slice_stats[fname]["ssim"].get(slice_idx, '') for fname in fnames]
                volume_slice_stats[f"PSNR_S{str(slice_idx + 1).zfill(2)}"] = [self.slice_stats[fname]["psnr"].get(slice_idx, '') for fname in fnames]

            pd.DataFrame.from_dict(volume_slice_stats, orient='columns').to_csv(Path(f'{self.experiments_path}', f'{self.folder_name}_volume_slice_stats.csv'), index=False)

            best_nmse = torch.mean(torch.tensor(([volume_stats[fname]["nmse"].item() for fname in fnames]))).item()
            best_psnr = torch.mean(torch.tensor(([volume_stats[fname]["psnr"].item() for fname in fnames]))).item()
            best_ssim = torch.mean(torch.tensor(([volume_stats[fname]["ssim"].item() for fname in fnames]))).item()
            logger.info(f'Best performance recorded >> epoch: {self.epoch_count} | NMSE: {best_nmse:.4f} | PSNR: {best_psnr:.2f} | SSIM: {best_ssim:.4f}')
            self.best_epoch = True

        torch.save({'epoch': self.epoch_count,
                    'last_model_state_dict': last_model_state_dict,
                    'last_optimizer_state_dict': last_optimizer_state_dict,
                    'best_model_state_dict': self.best_model_state_dict,
                    'best_val_loss': self.best_val_loss,
                    }, os.path.join(self.experiments_path, f'{self.folder_name}_model.pth'))

    def visualize(self, fnames, slice_nums, sequences, zfimages, outputs, targets, accs, max_values):

        for fname, slice_num, sequence, zfimage, output, target, acc, max_value in zip(fnames, slice_nums, sequences, zfimages, outputs, targets, accs, max_values):

            zfimage = zfimage.flip([1])
            output = output.flip([1])
            target = target.flip([1])

            if self.epoch_count % self.plot_freq == 0:
                fig_path = os.path.join(self.fig_path, f'epoch_{self.epoch_count}')
                os.mkdir(fig_path) if not os.path.isdir(fig_path) else None
                save_image(make_grid([zfimage, output, target], nrow=3, padding=0), os.path.join(fig_path, f'{fname}_{slice_num}_{acc}X.jpg'), normalize =True, value_range=(0, max_value))

            if self.best_epoch:
                fig_path = os.path.join(self.fig_path, f'epoch_best')
                os.mkdir(fig_path) if not os.path.isdir(fig_path) else None
                save_image(make_grid([zfimage, output, target], nrow=3, padding=0), os.path.join(fig_path, f'{fname}_{slice_num}_{acc}X.jpg'), normalize =True, value_range=(0, max_value))
                fig_subpath = os.path.join(fig_path, f'{fname}_{slice_num}_{acc}X')
                os.mkdir(fig_subpath) if not os.path.isdir(fig_subpath) else None
                save_image(make_grid([zfimage], nrow=1, padding=0), os.path.join(fig_subpath, 'ZF.jpg'), normalize =True, value_range=(0, max_value))
                save_image(make_grid([output], nrow=1, padding=0), os.path.join(fig_subpath, 'OUT.jpg'), normalize =True, value_range=(0, max_value))
                save_image(make_grid([target], nrow=1, padding=0), os.path.join(fig_subpath, 'REF.jpg'), normalize =True, value_range=(0, max_value))
