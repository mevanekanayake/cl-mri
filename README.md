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
 McSTRA
  ├── train_clmri.py
  ├── models
  │     └── clmri.py



  ├── utils
  │     ├── data.py
  │     ├── evaluate.py
  │     ├── fourier.py
  │     ├── manager.py
  │     ├── mask.py
  │     ├── math.py
  │     ├── mulitcoil.py
  │     └── transforms.py
  ├── paths.json
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
python train_mcstra.py --batch_size=8 --num_epochs=50 --tvsr=1. --vvsr=1. --num_hts=2 --embed_dims=48,96,48
```

## Train downstream reconstruction Models


## Train baseline reconstruction Models


## Acknowledgements
- This repository makes use of the fastMRI codebase for training, evaluation, and data preparation: https://github.com/facebookresearch/fastMRI


## Citing Our Work
Please cite: M. Ekanayake, Z. Chen, M. Harandi, G. Egan, and Z. Chen, CL-MRI: Self-Supervised Contrastive Learning to Improve the Accuracy of Undersampled MRI Reconstruction. 2024. https://arxiv.org/abs/2306.00530v2
