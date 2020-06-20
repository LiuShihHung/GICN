## 	Learning Gaussian Instance Segmentation in Point Clouds
Shih-Hung Liu, Shang Yi Yu, Shao-Chi Wu, Hwann-Tzong Chen, Tyng-Luh Liu
### (1) Setup
ubuntu 16.04 + cuda 10.0

python 2.7 or 3.6

pytorch 1.2.0

scipy 1.3

h5py 2.9

open3d-python 0.3.0

### (2) Data
S3DIS: we use the same data released by [JSIS3D](https://github.com/pqhieu/jsis3d).

ScnaNet: you can download the ScanNet data in [ScanNet](http://www.scan-net.org).


### (3) Train/test
python train.py

python eval.py

### (4) Quantitative Results on ScanNet

![Arch Image](https://github.com/LiuShihHung/GICN/figs/fig_ins_scannet.png)