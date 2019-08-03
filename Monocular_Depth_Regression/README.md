# Uncertainty Estimation for Unsupervised Monocular Depth Regression on KITTI

## General Background

We apply to a dropout-extended version of [Unsupervised Monocular Depth Regression](https://arxiv.org/abs/1609.03677). This code is based on the [code of the original work](https://github.com/mrharicot/monodepth). Our approximation results are expected to be better, due to the absence of a softmax activation. 

Our dropout-extended version simply places a dropout (p=0.1) prior to the last convolutional layer.

## How to use this code?

- Download KITTI if necessary by invoking
```shell
wget -i utils/kitti_archives_to_download.txt -P ~/my/output/folder/
```
- Train a dropout-extended network following the instruction of the [repository of the original work](https://github.com/mrharicot/monodepth). When running the trainings/test script **add the additional argument --use_dropout 1**. Alternatively checkpoints of a pretrained model may be downloaded from [here](https://drive.google.com/open?id=1Z-j9NM2UFqPW4etuOcjoyDIQBVaSk1qH)
- For uncertainty evaluation follow the notebook **eval_uncertainty.ipynb** 
