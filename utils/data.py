import torch
from torch.utils.data import Dataset

import os
import random
import numpy as np
import nibabel as nib
import h5py


class Data(Dataset):

    def __init__(self, root, train, seq_types, nv=0, transform=None, viz=False):

        self.root = root
        self.transform = transform
        self.key = "train" if train else "val"

        lib_path = os.path.join(root, "library.pt")
        self.library = torch.load(lib_path)
        self.seq_types = seq_types
        self.data_per_seq = ""
        self.examples = []
        self.selected_examples = []

        # COLLECT ALL SLICES
        # nv!= 0 means partial dataset. nv= 0 means full dataset.
        if nv != 0:
            vols_per_seq = int(nv / len(self.seq_types))
            self.num_volumes = vols_per_seq * len(self.seq_types)
            for seq_type in self.seq_types:
                seq_examples = [item for sublist in self.library[self.key][seq_type][:vols_per_seq] for item in sublist]
                self.examples += seq_examples
                if viz:
                    self.selected_examples += random.sample([item[0] for item in self.library[self.key][seq_type][:vols_per_seq]], k=viz)
                    # collect the file names of randomly selected viz amount of volumes from each sequence out of the sampled volumes
                else:
                    self.data_per_seq += f'{seq_type:<{len(max(seq_types, key=len))+1}}: {vols_per_seq} | {len(seq_examples)}\n'

        else:
            self.num_volumes = sum([len(self.library[self.key][seq_type]) for seq_type in self.seq_types])
            for seq_type in self.seq_types:
                seq_examples = [item for sublist in self.library[self.key][seq_type] for item in sublist]
                self.examples += seq_examples
                if viz:
                    self.selected_examples += random.sample([item[0] for item in self.library[self.key][seq_type]], k=viz)
                    # collect the file names of randomly selected viz amount of volumes from each sequence out of all volumes
                else:
                    self.data_per_seq += f'{seq_type:<{len(max(seq_types, key=len))+1}}: {len(self.library[self.key][seq_type])} | {len(seq_examples)}\n'

        # TO CREATE VIZ DATASET
        self.examples = self.selected_examples if viz else self.examples

    def __len__(self):
        return len(self.examples)

    # def __getitem__(self, i):
    #     fname, slice_id = self.examples[i]
    #     data = torch.load(os.path.join(self.root, self.key, f"{fname}.pt"))
    #     kspace = data["kspace"][slice_id]
    #     sequence = data["sequence"]
    #     sample = self.transform(kspace, fname, slice_id, sequence)

    def __getitem__(self, i):
        fname, slice_id = self.examples[i]
        data = torch.load(os.path.join(self.root,"multicoil_"+self.key, f"{fname}.pt"))
        kspace = data["kspace"][slice_id]
        sequence = data["sequence"]
        max_val = data["max_val"]
        sample = self.transform(kspace, fname, slice_id, sequence, max_val)

        return sample


class MBIData(Dataset):

    def __init__(self, root, transform=None):

        self.root = root
        self.transform = transform

        train_set = [f"sub-{str(i).zfill(2)}_T1w" for i in range(1,28)]
        train_set.remove('sub-27_T1w')
        train_set.remove('sub-21_T1w')
        self.examples = [(a, b) for a in train_set for b in range(140, 200)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice_id = self.examples[i]
        data = nib.load(os.path.join(self.root, f"{fname}.nii.gz"))
        img = np.array(data.dataobj)   
        img = img[:,:, slice_id]
        sequence = "T1w"
        sample = self.transform(img, fname, slice_id-140, sequence)

        return sample
    
    
class ZhifengData(Dataset):

    def __init__(self, root, transform=None):

        self.root = root
        self.transform = transform
        self.examples = []

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname = self.examples[i]

        with h5py.File(os.path.join(self.root, f"{fname}.mat"), 'r') as f:
            img = f["I_com"][:]
        
        img = np.stack((img['real'],img['imag']), axis=-1)
        slice_id = 0
        sequence = "T1w"
        sample = self.transform(img, fname, slice_id, sequence)

        return sample