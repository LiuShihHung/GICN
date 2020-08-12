#	Learning Gaussian Instance Segmentation in Point Clouds

## Introduction
![Arch Image](https://github.com/LiuShihHung/GICN/blob/master/figs/architecture.png)

Shih-Hung Liu, Shang Yi Yu, Shao-Chi Wu, Hwann-Tzong Chen, Tyng-Luh Liu
### (1) Setup
ubuntu 16.04 + cuda 10.1

python 3.6

pytorch 1.5.1

scipy 1.3

h5py 2.9

open3d-python 0.3.0

### (2) Data
S3DIS: we use the same data released by [JSIS3D](https://github.com/pqhieu/jsis3d). You can download the data into the ./data_s3dis

ScnaNet: you can download the ScanNet data in [ScanNet](http://www.scan-net.org).


### (3) Train/test

```bash

python train.py

python main_eval.py

```

### (4) Compilation

1. Compiling the pointnet++ module

```bash

cd Pointnet2.PyTorch/pointnet2

python setup.py install

```

2. You also need to compiling SCN for semantic prediction

The environment is based on [facebookresearch/SparseConvNet](https://github.com/facebookresearch/SparseConvNet)

### (5) Quantitative Results on ScanNet

![Arch Image](https://github.com/LiuShihHung/GICN/blob/master/figs/fig_ins_scannet.png)

### (6) Pre-trained model

The pretrained GICN on S3dis dataset is in ./experiment 

Evaluation on Area5:

-precision : 0.6348
 
-recall : 0.4669

### (7) Acknowledgements 

Pointnet++ is based on [sshaoshuai/Pointnet2.PyTorch](https://github.com/sshaoshuai/Pointnet2.PyTorch)