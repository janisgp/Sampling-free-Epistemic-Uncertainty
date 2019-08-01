# UCI Regression Datasets

## Background

We compare the predictive performance of our approach with MC dropout. The code of the experiment is based on the experiment in [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://arxiv.org/abs/1506.02142) and the original code can be found [here](https://github.com/yaringal/DropoutUncertaintyExps). The main goal of this experiment is to show quantitatively that our approximation delivers good results.

We follow the original setup. We split the training data 20 times randomly into training and validation set, except for the dataset Protein Structure where we use five splits, and perform a separate grid search for the hyperparameters dropout rate and tau. Following the original work we use one hidden layer with 50 hidden units, except for Protein Structure where we use 100 hidden units. Dropout is applied directly to the input and after the hidden layer and we train the network for 400 epochs. We use the full covariance matrix to propagate uncertainty.

The TLL in the original experiment requires sampling from distributions of outputs. Since our method naturally just returns the parameters of a unimodal distribution over the outputs, we assume a Gaussian distribution and sample from it to compute the TLL. According to [Fast Dropout](https://nlp.stanford.edu/pubs/sidaw13fast.pdf) this is a reasonable assumption given our hidden layer dimension. We perform the same grid search as for MC dropout. For both, MC dropout and our proposed approximation, we sample 10000 predictions to compute the TLL.

This code checks the performance of four distinct approaches: MC dropout, our sampling-free approximation, Mixture Density Networks and learning the dropout rate using our approach. The first two and part of our publication. 

The results can be found in our work(LINK!!!)

## How to use the code?

We describe the essential components of this code. We **keras** for our implementation. 

### net

This folder contains neural network implementations. Most importantly:
- net.py: implements initialization, training and testing of networks using MC dropout and our approximation. 
- net_mdn.py: implements initialization, training and testing of Mixture Density Networks.
- net_learn_drop_rate.py: implements initialization, training and testing for learning the dropout parameter using our sampling-free approximation.

### experiment.py

Runs the experiment. One ca pass a variety of arguments which are explained in the following:
- --dir: name of dataset directory
- --epochx: multiplicator for number of epochs
- --hidden: number of hidden layers
- --full_cov: whether or not (1/0) to use the full covariance matrix for propagating variance when applying our approximation
- --full_cov_drop: whether or not (1/0) to use the full covariance matrix for propagating variance when learning the dropout parameter
- --per_activation_rate: when learning dropout parameter -> learn one dropout rate per activation (1) or one dropout rate per layer (0)
- --gpu: which gpu to use
- --dropout: whether or not to use MC dropout (1/0)
- --mdn: whether or not to use Mixture Density Networks (1/0)
- --learn_dropout: whether or not (1/0) to learn the dropout parameter
- --learning_rate: learning rate
- --initial_drop_rate: when learning the dropout rate -> initial value

To run the experiments in our publication, one has to parse the following arguments:

**--dir dataset_name --epochx 10 --hidden 1 --dropout 1 --full_cov 1**

### Uncertainty_Propagator

This code implements a frame for automatically adjusting a keras model for sampling-free uncertainty propagation. This code is not applicable to general keras models as it only implements a subset of layers and only supports sequential models. 
