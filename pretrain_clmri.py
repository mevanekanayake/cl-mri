import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm
import argparse
from pathlib import Path
from collections import OrderedDict
import pandas as pd
import os

from utils.data import Data
from utils.transform import Transform_CLR
from utils.manager import set_seed, set_cuda, fetch_paths, set_logger, set_device

from models.varnet import VarNet
from losses.supconloss import SupConLoss
import matplotlib.pyplot as plt

def forward_pass_pretrain_varnet_clr(positive_samples, model, loss_function, args):

    features = []
    for batch in positive_samples:

        kspace_und = batch.kspace_und.to(args.dv)
        mask = batch.mask.to(args.dv)
        num_low_freqs = batch.num_low_freqs.to(args.dv)
        out = model(masked_kspace=kspace_und,
                    mask=mask,
                    num_low_frequencies=num_low_freqs)
        # .unsqueeze(1)
        features.append(out)

    features = torch.stack(features, dim=1)
    loss = loss_function(features)
    
    return loss


def train_():
    # SET ARGUMENTS
    parser = argparse.ArgumentParser()

    # DATA ARGS
    parser.add_argument("--trainacc", type=str, default="2,4,6,8", help="Acceleration factors for the k-space undersampling")
    parser.add_argument("--valacc", type=str, default="2,4,6,8", help="Acceleration factors for the k-space undersampling")
    parser.add_argument("--tnv", type=int, default=2, help="Number of volumes used for training [set to 0 for the full dataset]")
    parser.add_argument("--vnv", type=int, default=2, help="Number of volumes used for validation [set to 0 for the full dataset]")
    parser.add_argument("--viznv", type=int, default=1, help="Number of slices per sequence to visualize")
    parser.add_argument("--mtype", type=str, default="random", choices=("random", "equispaced"), help="Type of k-space mask")
    parser.add_argument("--dset", type=str, default="fastmribrain", choices=("fastmriknee", "fastmribrain"), help="Which dataset to use")
    parser.add_argument("--seq_types", type=str, default="AXFLAIR", help="Which sequence types to use")

    # TRAIN ARGS
    parser.add_argument("--bs", type=int, default=2, help="Batch size for training and validation")
    parser.add_argument("--ne", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--dv", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for model training")
    parser.add_argument("--dp", type=str, default=None, help="Whether to perform Data parallelism")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers")

    # EXPERIMENT ARGS
    parser.add_argument("--ckpt", type=str, help="Continue trainings from checkpoint")
    parser.add_argument("--pf", type=int, default=10, help="Plotting frequency")

    # MODEL ARGS
    parser.add_argument("--num_cascades", type=int, default=2, help="number of unrolled iterations") #12
    parser.add_argument("--pools", type=int, default=2, help="number of pooling layers for U-Net") #4
    parser.add_argument("--chans", type=int, default=9, help="number of top-level channels for U-Net") #18
    parser.add_argument("--sens_pools", type=int, default=2, help="number of pooling layers for sense est. U-Net") #4
    parser.add_argument("--sens_chans", type=int, default=4, help="number of top-level channels for sense est. U-Net") #8

    # LOAD ARGUMENTS
    args = parser.parse_args()
    args.seq_types = args.seq_types.split(',')
    args.trainacc = [int(accel) for accel in args.trainacc.split(',')]
    args.valacc = [int(accel) for accel in args.valacc.split(',')]

    # LOAD CHECKPOINT
    ckpt = torch.load(Path(args.ckpt), map_location='cpu') if args.ckpt else None
    args.ne = args.ne - ckpt['epoch'] if args.ckpt else args.ne

    # SET SEED
    set_seed()

    # SET CUDA
    set_cuda()

    # SET/CREATE PATHS
    data_path, exp_path = fetch_paths(args.dset)

    # LOG ARGS, PATHS
    logger = set_logger(exp_path)
    for entry in vars(args):
        logger.info(f'{entry}: {vars(args)[entry]}')
    logger.info(f'data_path = {str(data_path)}')
    logger.info(f'experiment_path = {str(exp_path)}')

    # LOAD MODEL
    model = VarNet(num_cascades = args.num_cascades,
                   sens_chans= args.sens_chans,
                   sens_pools= args.sens_pools,
                   chans=args.chans,
                   pools= args.pools,
    )

    model.load_state_dict(ckpt['last_model_state_dict']) if args.ckpt else None
    best_model_state_dict = ckpt['best_model_state_dict'] if ckpt else None
    best_val_loss = ckpt['best_val_loss'] if ckpt else float('inf')
    epoch_count = ckpt['epoch'] if ckpt else 0    
    
    logger.info(f'No. of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    # SET GPUS
    model, args, device_ids = set_device(model, args)
    logger.info(f'num GPUs available: {torch.cuda.device_count()}')
    logger.info(f'num GPUs using: {len(device_ids)}')
    logger.info(f'GPU model: {torch.cuda.get_device_name(args.dv)}') if torch.cuda.device_count() > 0 else None

    # LOAD TRAINING DATA
    train_transform = Transform_CLR(train=True, mask_type=args.mtype, accelerations=args.trainacc)
    train_dataset = Data(root=data_path, train=True, seq_types=args.seq_types, transform=train_transform, nv=args.tnv)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.bs, num_workers=args.num_workers, shuffle=True, pin_memory=True)
    logger.info(f'Training set: No. of volumes: {train_dataset.num_volumes} | No. of slices: {len(train_dataset)}')
    logger.info(f'{train_dataset.data_per_seq[:-1]}')

    # LOAD VALIDATION DATA
    val_transform = Transform_CLR(train=False, mask_type=args.mtype, accelerations=args.valacc)
    val_dataset = Data(root=data_path, train=False, seq_types=args.seq_types, transform=val_transform, nv=args.vnv)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.bs, num_workers=args.num_workers, shuffle=False, pin_memory=True)
    logger.info(f'Validation set: No. of volumes: {val_dataset.num_volumes} | No. of slices: {len(val_dataset)}')
    logger.info(f'{val_dataset.data_per_seq[:-1]}')

    # LOAD VISUALIZATION DATA
    viz_dataset = Data(root=data_path, train=False, seq_types=args.seq_types, transform=val_transform, nv=args.vnv, viz=args.viznv)
    viz_loader = DataLoader(dataset=viz_dataset, batch_size=args.bs, num_workers=args.num_workers, shuffle=False, pin_memory=True)

    # LOSS FUNCTION
    loss_fn = SupConLoss()

    # SET OPTIMIZER
    logger.info(f'Optimizer: RMSprop')
    optimizer = torch.optim.RMSprop(params=model.parameters(), lr=args.lr)
    optimizer.load_state_dict(ckpt['last_optimizer_state_dict']) if args.ckpt else None

    summary = OrderedDict({"epoch_no": [],
                           "train_loss": [],
                           "val_loss": [],
                           })

    # LOOP
    for _ in range(args.ne):

        epoch_count += 1

        # BEGIN TRAINING LOOP
        model.train()
        train_loss = 0
        batch_count = 0
        with tqdm(train_loader, unit="batch") as train_epoch:

            for b in train_epoch:
                train_epoch.set_description(f"Epoch {epoch_count} [Training]")

                optimizer.zero_grad()

                loss = forward_pass_pretrain_varnet_clr(b, model, loss_fn, args)

                loss.backward()
                optimizer.step()

                # END TRAINING STEP
                train_epoch.set_postfix(train_loss=loss.detach().item())
                train_loss += loss.detach().to('cpu') * b[0].kspace_und.shape[0]
                batch_count += b[0].kspace_und.shape[0]

            epoch_train_loss = train_loss / batch_count

        model.eval()
        val_loss = 0
        batch_count = 0
        with torch.no_grad():

            # BEGIN VALIDATION LOOP
            with tqdm(val_loader, unit="batch") as val_epoch:
                for b in val_epoch:
                    val_epoch.set_description(f"Epoch {epoch_count} [Validation]")

                    loss = forward_pass_pretrain_varnet_clr(b, model, loss_fn, args)

                    # END VALIDATION STEP
                    val_epoch.set_postfix(val_loss=loss.detach().item())
                    val_loss += loss.detach().to('cpu') * b[0].kspace_und.shape[0]
                    batch_count += b[0].kspace_und.shape[0]

            epoch_val_loss = val_loss / batch_count

        # END EPOCH
        summary["epoch_no"].append(epoch_count)
        summary["train_loss"].append(float(epoch_train_loss))
        summary["val_loss"].append(float(epoch_val_loss))
        pd.DataFrame.from_dict(summary, orient='columns').to_csv(Path(os.path.join(f'{exp_path}', f'{exp_path.name}_summary.csv')), index=False)

        last_model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        last_optimizer_state_dict = optimizer.state_dict()

        if epoch_val_loss <= best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state_dict = last_model_state_dict

        torch.save({'epoch': epoch_count,
                    'last_model_state_dict': last_model_state_dict,
                    'last_optimizer_state_dict': last_optimizer_state_dict,
                    'best_model_state_dict': best_model_state_dict,
                    'best_val_loss': best_val_loss,
                    }, os.path.join(exp_path, f'{exp_path.name}_model.pth'))
        

if __name__ == '__main__':
    train_()
    print('Done!')
