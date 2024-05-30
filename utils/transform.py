import torch

import re
from typing import NamedTuple
import random

from utils.fourier import ifft2c as ift
from utils.math import complex_abs, rss, normalize_instance, normalize
from utils.mask import apply_random_mask, apply_equispaced_mask, apply_random_wo_cent_mask, apply_equispaced_wo_cent_mask


class Sample(NamedTuple):
    kspace: torch.Tensor
    kspace_und: torch.Tensor
    mask: torch.Tensor
    image_zf: torch.Tensor
    image_zf2: torch.Tensor
    target: torch.Tensor
    fname: str
    slice_num: int
    sequence: str
    acceleration: int
    num_low_freqs: int
    max_val: float
    

class Transform:

    def __init__(self, train, mask_type, accelerations):
        self.train = train
        self.mask_type = mask_type
        self.accelerations = accelerations

    def __call__(self, kspace_ori, fname, slice_num, sequence, max_val):

        target = rss(complex_abs((ift(kspace_ori))))

        seed = int("".join(re.findall(r"\d+", fname))) if not self.train else None
        torch.manual_seed(seed) if seed else None
        choice = torch.randint(0, len(self.accelerations), (1,))
        acceleration = self.accelerations[choice]
        center_fraction = 0.32 / acceleration

        if self.mask_type == 'random':
            kspace_und, mask, acc, nlf = apply_random_mask(kspace_ori, acceleration, center_fraction)
        elif self.mask_type == 'equispaced':
            kspace_und, mask, acc, nlf = apply_equispaced_mask(kspace_ori, acceleration, center_fraction)
        elif self.mask_type == 'random_wo_cent':
            kspace_und, mask, acc = apply_random_wo_cent_mask(kspace_ori, acceleration)
        elif self.mask_type == 'equispaced_wo_cent':
            kspace_und, mask, acc = apply_equispaced_wo_cent_mask(kspace_ori, acceleration)
        else:
            raise ValueError("This code base currently accomodates only random masking.")

        mask = (mask==True).unsqueeze(-1).repeat(1,1,2).unsqueeze(0).repeat(kspace_ori.shape[0],1,1,1)

        image_zf2 = ift(kspace_und)
        image_zf = complex_abs(image_zf2)

        sample = Sample(
            kspace=kspace_ori,
            kspace_und=kspace_und,
            mask=mask,
            image_zf=image_zf.unsqueeze(1),
            image_zf2=image_zf2,
            target=target.unsqueeze(0),
            fname=fname,
            slice_num=slice_num,
            sequence=sequence,
            acceleration=acc,
            num_low_freqs=nlf,
            max_val=max_val,
        )

        return sample
    

class Transform_Noise:

    def __init__(self, train, mask_type, accelerations, db):
        self.train = train
        self.mask_type = mask_type
        self.accelerations = accelerations
        self.db = db

    def __call__(self, kspace_ori, fname, slice_num, sequence, max_val):

        target = rss(complex_abs((ift(kspace_ori))))

        SNRdB = self.db
        gamma = 10 ** (SNRdB / 10)  # SNR to linear scale
        kspace_abs = complex_abs(kspace_ori)
        P = torch.sum(torch.sum(kspace_abs ** 2)) / torch.numel(kspace_abs)
        N0 = P / gamma  # Find the noise spectral density
        n = torch.sqrt(N0 / 2) * (torch.randn(kspace_ori.shape))
        kspace_ori = kspace_ori + n  # received signal

        seed = int("".join(re.findall(r"\d+", fname))) if not self.train else None
        torch.manual_seed(seed) if seed else None
        choice = torch.randint(0, len(self.accelerations), (1,))
        acceleration = self.accelerations[choice]
        center_fraction = 0.32 / acceleration

        if self.mask_type == 'random':
            kspace_und, mask, acc, nlf = apply_random_mask(kspace_ori, acceleration, center_fraction)
        elif self.mask_type == 'equispaced':
            kspace_und, mask, acc, nlf = apply_equispaced_mask(kspace_ori, acceleration, center_fraction)
        elif self.mask_type == 'random_wo_cent':
            kspace_und, mask, acc = apply_random_wo_cent_mask(kspace_ori, acceleration)
        elif self.mask_type == 'equispaced_wo_cent':
            kspace_und, mask, acc = apply_equispaced_wo_cent_mask(kspace_ori, acceleration)
        else:
            raise ValueError("This code base currently accomodates only random masking.")

        mask = (mask==True).unsqueeze(-1).repeat(1,1,2).unsqueeze(0).repeat(kspace_ori.shape[0],1,1,1)

        image_zf2 = ift(kspace_und)
        image_zf = complex_abs(image_zf2)

        sample = Sample(
            kspace=kspace_ori,
            kspace_und=kspace_und,
            mask=mask,
            image_zf=image_zf.unsqueeze(1),
            image_zf2=image_zf2,
            target=target.unsqueeze(0),
            fname=fname,
            slice_num=slice_num,
            sequence=sequence,
            acceleration=acc,
            num_low_freqs=nlf,
            max_val=max_val,
        )

        return sample



class Sample_CLR(NamedTuple):
    kspace_und: torch.Tensor
    mask: torch.Tensor
    image_zf: torch.Tensor
    num_low_freqs: int


class Transform_CLR:

    def __init__(self, train, mask_type, accelerations):
        self.train = train
        self.mask_type = mask_type
        self.accelerations = accelerations

    def __call__(self, kspace_ori, fname, slice_num, sequence, max_val):

        target = rss(complex_abs((ift(kspace_ori))))

        seed = int("".join(re.findall(r"\d+", fname))) if not self.train else None
        torch.manual_seed(seed) if seed else None

        sample = []

        for acceleration in self.accelerations:

            center_fraction = 0.32 / acceleration

            if self.mask_type == 'random':
                kspace_und, mask, acc, nlf = apply_random_mask(kspace_ori, acceleration, center_fraction)
            elif self.mask_type == 'equispaced':
                kspace_und, mask, acc, nlf = apply_equispaced_mask(kspace_ori, acceleration, center_fraction)
            elif self.mask_type == 'random_wo_cent':
                kspace_und, mask, acc = apply_random_wo_cent_mask(kspace_ori, acceleration)
            elif self.mask_type == 'equispaced_wo_cent':
                kspace_und, mask, acc = apply_equispaced_wo_cent_mask(kspace_ori, acceleration)
            else:
                raise ValueError("This code base currently accomodates only random masking.")

            mask = (mask==True).unsqueeze(-1).repeat(1,1,2).unsqueeze(0).repeat(kspace_ori.shape[0],1,1,1)

            image_zf2 = ift(kspace_und)
            image_zf = rss(complex_abs(image_zf2))

            image_zf_normalized, mean, std = normalize_instance(image_zf, eps=1e-11)
            target_normalized = normalize(target, mean, std, eps=1e-11)

            sample.append(Sample_CLR(
            kspace_und=kspace_und,
            mask=mask,
            image_zf=image_zf.unsqueeze(0),
            num_low_freqs=nlf,
        ))

        return sample
    

class Sample_CLR_with_scan_details(NamedTuple):
    kspace_und: torch.Tensor
    mask: torch.Tensor
    image_zf: torch.Tensor
    num_low_freqs: int
    fname: str
    slice_num: int
    sequence: str
    max_val: float


class Transform_CLR_with_scan_details:

    def __init__(self, train, mask_type, accelerations):
        self.train = train
        self.mask_type = mask_type
        self.accelerations = accelerations

    def __call__(self, kspace_ori, fname, slice_num, sequence, max_val):

        target = rss(complex_abs((ift(kspace_ori))))

        seed = int("".join(re.findall(r"\d+", fname))) if not self.train else None
        torch.manual_seed(seed) if seed else None

        sample = []

        for acceleration in self.accelerations:

            center_fraction = 0.32 / acceleration

            if self.mask_type == 'random':
                kspace_und, mask, acc, nlf = apply_random_mask(kspace_ori, acceleration, center_fraction)
            elif self.mask_type == 'equispaced':
                kspace_und, mask, acc, nlf = apply_equispaced_mask(kspace_ori, acceleration, center_fraction)
            elif self.mask_type == 'random_wo_cent':
                kspace_und, mask, acc = apply_random_wo_cent_mask(kspace_ori, acceleration)
            elif self.mask_type == 'equispaced_wo_cent':
                kspace_und, mask, acc = apply_equispaced_wo_cent_mask(kspace_ori, acceleration)
            else:
                raise ValueError("This code base currently accomodates only random masking.")

            mask = (mask==True).unsqueeze(-1).repeat(1,1,2).unsqueeze(0).repeat(kspace_ori.shape[0],1,1,1)

            image_zf2 = ift(kspace_und)
            image_zf = rss(complex_abs(image_zf2))

            image_zf_normalized, mean, std = normalize_instance(image_zf, eps=1e-11)
            target_normalized = normalize(target, mean, std, eps=1e-11)

            sample.append(Sample_CLR_with_scan_details(
            kspace_und=kspace_und,
            mask=mask,
            image_zf=image_zf.unsqueeze(0),
            num_low_freqs=nlf,
            fname = fname,
            slice_num = slice_num,
            sequence = sequence,
            max_val = max_val
        ))

        return sample
