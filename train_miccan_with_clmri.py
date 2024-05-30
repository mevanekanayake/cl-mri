import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm
import argparse
from pathlib import Path

from utils.data import Data
from utils.transform import Transform
from utils.manager import set_seed, set_cuda, fetch_paths, set_logger, set_device, RunManager
from utils.math import batch_chans_to_chan_dim, chans_to_batch_dim, rss, complex_abs

from models.miccan import MICCAN
from models.varnet import VarNet
import matplotlib.pyplot as plt

def forward_pass_varnet_pretrained_miccan_lineval(batch, model, loss_function, args):

    kspace_und = batch.kspace_und.to(args.dv)
    mask = batch.mask.to(args.dv)
    num_low_freqs = batch.num_low_freqs.to(args.dv)

    _, out_miccan = model(masked_kspace=kspace_und,
                                            mask=mask,
                                            num_low_frequencies=num_low_freqs
                                            )
    
    tar = batch.target.to(args.dv)  # (b, c, h, w)
    loss = loss_function(tar, out_miccan)

    return loss, out_miccan


class Varnet_pretrained_MICCAN_lineval(nn.Module):
    def __init__(
            self, num_chans=256
    ):
        super().__init__()

        self.varnet_pretrained = VarNet(num_cascades=12,
                                        pools=4,
                                        chans=18,
                                        sens_pools=4,
                                        sens_chans=8,
                                        )
        self.miccan = MICCAN(block='UCA',
                 n_layer=5,
                 in_channel=2,
                 out_channel=2
    )

    def forward(self, masked_kspace, mask, num_low_frequencies):

        image = self.varnet_pretrained(masked_kspace, mask, num_low_frequencies) # (b, n_coils, h, w, c=2)
        out1 = complex_abs(image).unsqueeze(-1)    # (b, n_coils, h, w, c=1)
        out1_rss = rss(out1, dim=1).permute(0, 3, 1, 2)  # (b, c=1, h, w)

        image, b_image = chans_to_batch_dim(image) # (b*n_coils, 1, h, w, c=2)
        image = image.squeeze(1).permute(0, 3, 1, 2) # (b*n_coils, c=2, h, w)

        kspace_und, b_kspace = chans_to_batch_dim(masked_kspace) # (b*n_coils, 1, h, w, c=2)
        kspace_und = kspace_und.squeeze(1).permute(0, 3, 1, 2) # (b*n_coils, h, w, c=2)
    
        mask = mask.to(torch.float) # (b, n_coils, h, w, c=2)
        mask, b_mask = chans_to_batch_dim(mask) # (b*n_coils, 1, h, w, c=2)
        mask = mask.squeeze(1).permute(0, 3, 1, 2) # (b*n_coils, c=2, h, w)

        out2 = self.miccan(image=image,
                    kspace=kspace_und,
                    mask=mask)

        out2 = complex_abs(out2.permute(0, 2, 3, 1)).unsqueeze(-1)
        out2 = out2.unsqueeze(1)
        out2 = batch_chans_to_chan_dim(out2, b_image).permute(0, 1, 4, 2, 3) # (b, n_coils, c=1, h, w)
        out2 = rss(out2, dim=1)   # (b, c=1, h, w)

        return out1_rss, out2



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
    parser.add_argument("--bs", type=int, default=1, help="Batch size for training and validation")
    parser.add_argument("--ne", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--dv", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for model training")
    parser.add_argument("--dp", type=str, default=None, help="Whether to perform Data parallelism")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers")

    # EXPERIMENT ARGS
    parser.add_argument("--ckpt", type=str, help="Continue trainings from checkpoint")
    parser.add_argument("--pret", type=str, help="Weights of the pretrianed VarNet")
    parser.add_argument("--pf", type=int, default=10, help="Plotting frequency")
    parser.add_argument("--num_chans", type=int, default=8, help="U-net size")

    # LOAD ARGUMENTS
    args = parser.parse_args()
    args.seq_types = args.seq_types.split(',')
    args.trainacc = [int(accel) for accel in args.trainacc.split(',')]
    args.valacc = [int(accel) for accel in args.valacc.split(',')]

    # LOAD CHECKPOINT
    if args.ckpt is None and args.pret is not None:
        ckpt = None
        pret = torch.load(Path(args.pret), map_location='cpu')
    elif args.ckpt is not None and args.pret is None:
        ckpt = torch.load(Path(args.ckpt), map_location='cpu')
        args.ne = args.ne - ckpt['epoch']
    else:
        raise ValueError('MICCAN checkpoint weights and pretrained VarNet weights both cannot be None or loaded at the same time!')

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
    model = Varnet_pretrained_MICCAN_lineval(num_chans = args.num_chans)

    # load pretrained VarNet weights and freeze
    model.varnet_pretrained.load_state_dict(pret['best_model_state_dict']) if args.pret else None
    for param in model.varnet_pretrained.parameters():
        param.requires_grad = False

    # load ckpt weights
    model.load_state_dict(ckpt['last_model_state_dict']) if args.ckpt else None

    # trainable params count
    logger.info(f'No. of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    # SET GPUS
    model, args, device_ids = set_device(model, args)
    logger.info(f'num GPUs available: {torch.cuda.device_count()}')
    logger.info(f'num GPUs using: {len(device_ids)}')
    logger.info(f'GPU model: {torch.cuda.get_device_name(args.dv)}') if torch.cuda.device_count() > 0 else None

    # LOAD TRAINING DATA
    train_transform = Transform(train=True, mask_type=args.mtype, accelerations=args.trainacc)
    train_dataset = Data(root=data_path, train=True, seq_types=args.seq_types, transform=train_transform, nv=args.tnv)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.bs, num_workers=args.num_workers, shuffle=True, pin_memory=True)
    logger.info(f'Training set: No. of volumes: {train_dataset.num_volumes} | No. of slices: {len(train_dataset)}')
    logger.info(f'{train_dataset.data_per_seq[:-1]}')

    # LOAD VALIDATION DATA
    val_transform = Transform(train=False, mask_type=args.mtype, accelerations=args.valacc)
    val_dataset = Data(root=data_path, train=False, seq_types=args.seq_types, transform=val_transform, nv=args.vnv)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.bs, num_workers=args.num_workers, shuffle=False, pin_memory=True)
    logger.info(f'Validation set: No. of volumes: {val_dataset.num_volumes} | No. of slices: {len(val_dataset)}')
    logger.info(f'{val_dataset.data_per_seq[:-1]}')

    # LOAD VISUALIZATION DATA
    viz_dataset = Data(root=data_path, train=False, seq_types=args.seq_types, transform=val_transform, nv=args.vnv, viz=args.viznv)
    viz_loader = DataLoader(dataset=viz_dataset, batch_size=args.bs, num_workers=args.num_workers, shuffle=False, pin_memory=True)

    # LOSS FUNCTION
    loss_fn = nn.L1Loss()

    # SET OPTIMIZER
    logger.info(f'Optimizer: RMSprop')
    optimizer = torch.optim.RMSprop(params=model.parameters(), lr=args.lr)
    optimizer.load_state_dict(ckpt['last_optimizer_state_dict']) if args.ckpt else None

    # INITIALIZE RUN MANAGER
    m = RunManager(exp_path, ckpt, args.seq_types, args.pf)

    # LOOP
    for _ in range(args.ne):
        # BEGIN EPOCH
        m.begin_epoch()

        # BEGIN TRAINING LOOP
        model.train()
        with tqdm(train_loader, unit="batch") as train_epoch:

            for b in train_epoch:
                train_epoch.set_description(f"Epoch {m.epoch_count} [Training]")

                optimizer.zero_grad()

                train_loss, _ = forward_pass_varnet_pretrained_miccan_lineval(b, model, loss_fn, args)

                train_loss.backward()
                optimizer.step()

                # END TRAINING STEP
                train_epoch.set_postfix(train_loss=train_loss.detach().item())
                m.end_train_step(train_loss.detach().to('cpu'), b[0].shape[0])

        model.eval()
        with torch.no_grad():

            # BEGIN VALIDATION LOOP
            with tqdm(val_loader, unit="batch") as val_epoch:
                for b in val_epoch:
                    val_epoch.set_description(f"Epoch {m.epoch_count} [Validation]")

                    val_loss, x_hat = forward_pass_varnet_pretrained_miccan_lineval(b, model, loss_fn, args)

                    # END VALIDATION STEP
                    val_epoch.set_postfix(val_loss=val_loss.detach().item())
                    m.end_val_step(b.fname, b.slice_num, b.sequence, rss(b.image_zf, dim=1), x_hat.to('cpu'), b.target, val_loss.to('cpu'), b.max_val)

            # END EPOCH
            m.end_epoch(model, optimizer, logger)

            # VISUALIZATION
            if m.best_epoch or (m.epoch_count % args.pf == 0):
                with tqdm(viz_loader, unit="batch") as viz_epoch:
                    for b in viz_epoch:
                        viz_epoch.set_description(f"Epoch {m.epoch_count} [Visualization]")

                        _, x_hat = forward_pass_varnet_pretrained_miccan_lineval(b, model, loss_fn, args)

                        # END VISUALIZATION STEP
                        m.visualize(b.fname, b.slice_num, b.sequence, rss(b.image_zf, dim=1), x_hat.to('cpu'), b.target, b.acceleration, b.max_val)


if __name__ == '__main__':
    train_()
    print('Done!')
