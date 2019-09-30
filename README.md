# Sampling-free Epistemic Uncertainty Estimation Using Approximated Variance Propagation

This repository provides the code for the ICCV'19 publication ["Sampling-free Epistemic Uncertainty Estimation Using Approximated Variance Propagation"](https://arxiv.org/abs/1908.00598).  We provide a sampling-free approach for estimating epistemic uncertainty when applying methods based on noise injection (e.g. stochastic regularization). Our approach is motivated by error propagation. We primarily compare our approach with [Monte-Carlo (MC) dropout](https://arxiv.org/abs/1703.02914) by approximating the sampling procdeure of the latter. 

Following the experiment section in our paper, this repository is divided into three sections:
- **UCI_regression**: Comparison of predictive performance between MC dropout and our variance propagation approach. The code further implements [Mixture Density Networks](https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf) and learning the dropout parameter using our sampling-free approximation, which is not included in our publication. 
- **Bayesian_SegNet**: Sampling-free approximation of [Bayesian SegNet](https://arxiv.org/abs/1511.02680). We investigate the quality of our approximation on a high-dimensional semantic segmentation task. 
- **Monocular_Depth_Regression**: Applies our approximation to [Unsupervised Monocular Depth Regression](https://arxiv.org/abs/1609.03677). We enhance the original work by inserting dropout at the final layers and analyze the quality of our approximation. 
