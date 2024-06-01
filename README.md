# CL-MRI: Self-Supervised Contrastive Learning to Improve the Accuracy of Undersampled MRI Reconstruction

```
This is the official Pytorch implementation of the CL-MRI: Contrastive Learning MRI

```
![](imgs/model.png?raw=true)

## Environment
```
Prepare a python=3.8 environment and install the following python libraries.
1. pytorch==1.10.0
2. torchvision==0.11.1
3. numpy
4. h5py
5. pandas
6. scikit-image
7. tqdm
8. einops
9. timm

Alternatively, you can run the command "pip install -r requirements.txt" to install all the required libraries directly.
```

## Data Preparation
```
- For experiments we used the fastMRI knee and brain datasets (https://fastmri.med.nyu.edu/).

- The intensity values of the data were scaled in the range [0,1] as preprocessing.
```


## File structure

```
 cl-mri
  ├── pretrain_clmri.py
  ├── train_unet.py
  ├── train_unet_with_clmri.py
  ├── train_d5c5.py
  ├── train_d5c5_with_clmri.py
  ├── train_miccan.py
  ├── train_miccan_with_clmri.py
  ├── train_reconformer.py
  ├── train_reconformer_with_clmri.py
  ├── models
  │     ├── varnet.py
  │     ├── unet.py
  │     ├── d5c5.p
  │     ├── miccan.py
  │     └── reconformer.py
  ├── utils
  │     ├── data.py
  │     ├── evaluate.py
  │     ├── fourier.py
  │     ├── manager.py
  │     ├── mask.py
  │     ├── math.py
  │     ├── mulitcoil.py
  |     ├── paths.json
  │     └── transforms.py
  ├── losses
  │     └── supconloss.py
  ├── requirements.txt
  ├── data
  │     ├── knee
  │     └── brain
  ├── experiments
  │     ├── Experiment#1
  │     ├── Experiment#2
  │     :
  :     
    
```


## Train Pretraining model

```
python pretrain_clmri.py --seq_types=AXFLAIR --dp=2 --bs=4 --ne=100 --tnv=0 --vnv=0 --viznv=5 --num_cascades=12 --pools=4 --chans=18 --sens_pools=4 --sens_chans=8
```

## Train downstream reconstruction Models

D5C5
```
python train_d5c5_with_clmri.py --dp=0,1 --bs=2 --ne=100 --tnv=0 --vnv=0 --viznv=5 --lr=0.0001 --pret=<path to the trained contrastive featrure extractor>
```
U-Net
```
python train_unet_with_clmri.py --dp=0,1 --bs=2 --ne=100 --tnv=0 --vnv=0 --viznv=5 --lr=0.0001 --pret=<path to the trained contrastive featrure extractor>
```
MICCAN
```
python train_miccan_with_clmri.py --dp=0,1 --bs=2 --ne=100 --tnv=0 --vnv=0 --viznv=5 --lr=0.0001 --pret=<path to the trained contrastive featrure extractor>
```
ReconFormer
```
python train_reconformer_with_clmri.py --dp=0,1 --bs=2 --ne=100 --tnv=0 --vnv=0 --viznv=5 --lr=0.0001 --pret=<path to the trained contrastive featrure extractor>
```

## Train baseline reconstruction Models

D5C5
```
python train_d5c5.py --dp=0,1 --bs=2 --ne=100 --tnv=0 --vnv=0 --viznv=5 --lr=0.0001
```
U-Net
```
python train_unet.py --dp=0,1 --bs=2 --ne=100 --tnv=0 --vnv=0 --viznv=5 --lr=0.0001
```
MICCAN
```
python train_miccan.py --dp=0,1 --bs=2 --ne=100 --tnv=0 --vnv=0 --viznv=5 --lr=0.0001
```
ReconFormer
```
python train_reconformer.py --dp=0,1 --bs=2 --ne=100 --tnv=0 --vnv=0 --viznv=5 --lr=0.0001
```

## Acknowledgements
- This repository makes use of the fastMRI codebase for training, evaluation, and data preparation: https://github.com/facebookresearch/fastMRI


## Citing Our Work
Please cite: M. Ekanayake, Z. Chen, M. Harandi, G. Egan, and Z. Chen, CL-MRI: Self-Supervised Contrastive Learning to Improve the Accuracy of Undersampled MRI Reconstruction. 2024. https://arxiv.org/abs/2306.00530v3
