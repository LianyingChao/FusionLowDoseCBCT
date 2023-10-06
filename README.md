# Joint denoising and interpolation network for fusion low-dose CBCT

### JDINet architecture

![image](https://github.com/LianyingChao/FusionLowDoseCBCT/blob/master/figures/1.png)

### Requirement

Pytorch = 1.9.0 Tensorflow=1.15 Astra=1.9.9

### Load noisy data

Walnuts #1-21: https://zenodo.org/record/3763412;  put the noisy and sparse-view projs into ./ld_proj

### Load the trained model of JDINet and PostNet

JDINet/denoising and interpolation module: https://pan.baidu.com/s/1pjlDlRAYweXKySwrAfKrXg?pwd=nnix, put them into ./JDINet/saved_model

PostNet: https://pan.baidu.com/s/1s34dNINobnIWi9cEBSjg0A?pwd=2hbrr, put it into ./PostNet/Checkpoints

### Dual-processing for improving the quality of CBCT, e.g, 22-fold low dose

1.11-fold low-intensity and 2-fold sparse-view projs (P1,P3,P5,...,P499) are prepared to ./ld_proj

2.Preprocessing: denoise and interpolate to noise-free and full-view projs (python ./JDINet/test_denoi.py & python ./JDINet/test_inter.py)

3.Reconstruction to preprocessed CBCT: python ./fdk.py

4.Postprocessing: further improving preprocessed CBCT

### Results

![image](https://github.com/LianyingChao/FusionLowDoseCBCT/blob/master/figures/2.png)

### Contact

chaolianying@hust.edu.cn

