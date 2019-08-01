# Approximate sampling of Bayesian SegNet

In this experiment we approximate the sampling for [Bayesian SegNet](https://arxiv.org/abs/1511.02680) on a the task of semantic segmentation. We do so on the CamVid dataset. The keras implementation of Bayesian SegNet used in this experiment builds upon [this repository](https://github.com/Kautenja/semantic-segmentation-baselines). 

This experiment aims at showing qualitatively that our approximation scales to high-dimensional image data. Due to the size of the network and, most importantly, the softmax activation at the end of the network, absolut errors are rather large. However, a calibration plot shows, that one may still use our approximation, since its qualitative behaviour is similar to the uncertainty estimate obtained by the original MC dropout.
