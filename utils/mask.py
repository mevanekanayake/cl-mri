import torch
import numpy as np

# RANDOM MASK
def apply_random_mask(kspace, acceleration, center_fraction):
    num_rows, num_cols = kspace.shape[1], kspace.shape[2]
    num_low_freqs = int(round(num_cols * center_fraction))
    prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)
    mask = torch.rand(num_cols) < prob
    pad = (num_cols - num_low_freqs + 1) // 2
    mask[pad: pad + num_low_freqs] = True
    mask = mask.repeat(num_rows, 1).long()
    masked_kspace = kspace * mask.unsqueeze(0).unsqueeze(-1)

    return masked_kspace, mask, acceleration, num_low_freqs

# EQUISPACED MASK
def apply_equispaced_mask(kspace, acceleration, center_fraction):
    num_rows, num_cols = kspace.shape[1], kspace.shape[2]
    num_low_freqs = int(round(num_cols * center_fraction))
    
    prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)
    mask = torch.zeros(num_cols)
    mask[torch.linspace(0, num_cols-1, round(prob*num_cols), dtype=int)] = True
    pad = (num_cols - num_low_freqs + 1) // 2
    mask[pad: pad + num_low_freqs] = True
    mask = mask.repeat(num_rows, 1).long()
    masked_kspace = kspace * mask.unsqueeze(0).unsqueeze(-1)

    return masked_kspace, mask, acceleration, num_low_freqs

def apply_random_wo_cent_mask(kspace, acceleration):

    mask = torch.zeros(kspace.shape[0], kspace.shape[1]).long()
    mask[:, np.random.choice(np.arange(320), size=int(320/acceleration), replace=False)] = 1.0
    masked_kspace = kspace * mask.unsqueeze(-1)

    return masked_kspace, mask, acceleration


def apply_equispaced_wo_cent_mask(kspace, acceleration):
    
    mask = torch.zeros(kspace.shape[0], kspace.shape[1]).long()
    mask[:, torch.arange(0, 320, acceleration)] = 1.0
    masked_kspace = kspace * mask.unsqueeze(-1)


    return masked_kspace, mask, acceleration


if __name__ == "__main__":
    kspace_ori = torch.randn(320, 320, 2)
    kspace_und, m, acc = apply_equispaced_mask(kspace_ori, 8, 0.04)
    print()