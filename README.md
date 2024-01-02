# RCDNet: A Model-driven Deep Neural Network  for Single Image Rain Removal (CVPR2020)
 
[Hong Wang](https://hongwang01.github.io/), Qi Xie, Qian Zhao, and [Deyu Meng](http://gr.xjtu.edu.cn/web/dymeng) [[PDF]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_A_Model-Driven_Deep_Neural_Network_for_Single_Image_Rain_Removal_CVPR_2020_paper.pdf) [[Supplementary Material]](http://openaccess.thecvf.com/content_CVPR_2020/supplemental/Wang_A_Model-Driven_Deep_CVPR_2020_supplemental.pdf) 

**The extension of this work is released as [DRCDNet](https://github.com/hongwang01/DRCDNet) where we propose a dynamic rain kernel inference mechanism.**

**This is a simple coding framework, which has better compatibility with running environments. The original coding framework for RCDNet CVPR2020 is released at https://github.com/hongwang01/RCDNet**


## Dependicies

This repository is tested under the following system settings:

Python 3.6

Pytorch 1.4.0

CUDA 10.1

GPU NVIDIA Tesla V100-SMX2


## Dataset

Download Rain100L (training data: train/small/, testing data: test/small), Rain100H (training data: train/small/, testing data: test/small), Rain1400 (training data: train/small/, testing data: test/small), SPA-Data (testing data: test/small) from the  [[NetDisk]](https://pan.baidu.com/s/1yV4ih7C4Xg0iazqSBB-U1Q) (pwd:uz8h) and put them into the folder "data" as:

```
data/syndata/Rain100L/train/small/rain
data/syndata/Rain100L/train/small/norain
data/syndata/Rain100H/train/small/rain
data/syndata/Rain100H/train/small/norain
data/syndata/Rain1400/train/small/rain
data/syndata/Rain1400/train/small/norain
data/spa-data/real_world"
data/spa-data/real_world_gt"
data/spa-data/real_world.txt"
data/spa-data/test/small/rain
data/spa-data/test/small/norain
```

## Training

1. Training on SynData

```
python train_main_syn.py --data_path data/syndata/Rain100L/small/rain --gt_path data/syndata/Rain100L/small/norain --log_dir synlogs --model_dir synmodels --gpu_id 0
```

2. Training on SPA-Data

```
python train_main_real.py --data_path data/spa-data/ --log_dir spalogs --model_dir spamodels --gpu_id 0
```

## Pretrained_Model and Derained_Results

Average PSNR/SSIM values on four datasets:

Dataset    | [RCDNet(CVPR2020)](https://github.com/hongwang01/RCDNet) |RCDNet_simplified    
-----------|-----------|-----------
Rain100L   |40.00/0.9860|39.73/0.9856
Rain100H   |31.28/0.9093|31.32/0.9095
Rain1400   |33.04/0.9472|33.09/0.9491
SPA-Data   |41.47/0.9834|41.52/0.9849

Please note that for the simplified framework, we currently have only experimental result records and pretrained models on Rain100L and Rain100H on hand.  Sorry for the inconvenience.  Currently, all the available resource can be downloaded from [NetDisk](https://pan.baidu.com/s/1VHI-ZsLZybdbZp5TIUAcnw?pwd=rcds)(rcds)

## To Do List
Release pretrained models on Rain1400 and SPA-Data

## Citation
```
@InProceedings{Wang_2020_CVPR,  
author = {Wang, Hong and Xie, Qi and Zhao, Qian and Meng, Deyu},  
title = {A Model-Driven Deep Neural Network for Single Image Rain Removal},  
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},  
month = {June},  
year = {2020}  
}
```

## Contact
If you have any question, please feel free to concat Hong Wang (Email: hongwang9209@hotmail.com)
