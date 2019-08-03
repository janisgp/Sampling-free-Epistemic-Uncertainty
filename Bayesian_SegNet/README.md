# Approximate sampling of Bayesian SegNet

## Background

In this experiment we approximate the sampling for [Bayesian SegNet](https://arxiv.org/abs/1511.02680) on a the task of semantic segmentation. We do so on the CamVid dataset. The keras implementation of Bayesian SegNet used in this experiment builds upon [this repository](https://github.com/Kautenja/semantic-segmentation-baselines) of Christian Kauten.

This experiment aims at showing qualitatively that our approximation scales to high-dimensional image data. Due to the size of the network and, most importantly, the softmax activation at the end of the network, absolut errors are rather large. However, a calibration plot shows, that one may still use our approximation, since its qualitative behaviour is similar to the uncertainty estimate obtained by the original MC dropout.

We train with batch size 8 and stochastic gradient descent with initial learning rate $0.1$ and exponential learning rate decay with base $0.95$ for 200 epochs. We use early stopping (watching the validation loss) with patience 50. 
The original images in CamVid have resolution 720x960 and 32 classes. Following \cite{Kendall2015BayesianSM} we only use 11 generalized classes and downsample the images to 360x480.

## How to use the code?

We recommend creating a separate environment and installing **requirements.txt**. One can download a **pretrained Bayesian SegNet** models with dropout after the four central blocks or only prior to the final layer from [here](https://drive.google.com/open?id=1XMwyWib9aO8dqZFVWkZLvfz9DuqS2LVp).

Subsequently one can simply follow the notebook **SegNet-CamVid11-Bayesian.ipynb**. This will ultimately create a calibration plot in our work and qualitative examples for MC dropout and our approximation.
