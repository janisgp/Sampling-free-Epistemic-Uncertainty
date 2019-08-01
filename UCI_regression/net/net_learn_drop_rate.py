# Copyright 2016, Yarin Gal, All rights reserved.
# This code is based on the code by Jose Miguel Hernandez-Lobato used for his
# paper "Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks".

import warnings

warnings.filterwarnings("ignore")

import time
from scipy.misc import logsumexp
import numpy as np

import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Input
from tensorflow.keras.layers import Dropout, Dense, Lambda
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python import assert_non_negative

from Uncertainty_Propagator.layers.affine_layers import DenseVarPropagationLayer, PreActivationLayer
from Uncertainty_Propagator.layers.activations import ReLUActivationVarPropagationLayer
from net.losses import learn_dropout_loss
from net.layers import LearnDropoutVarPropagationLayer, AssertionLayer, AddHomoscedasticUncertaintyLayer
from net.regularizer import GaussianPriorRegularizer as gpreg
from net.base_net import BaseNet


T = 10000


class net_learn_drop_rate(BaseNet):

    def __init__(self, X_train, y_train, X_val, y_val, n_hidden, n_epochs=40,
                 tau=1.0, dropout_rate: float=0.5,
                 use_cov: bool=False, per_activation_rate: bool=False,
                 regularizer: str='gr', debug: bool=False, lr: float=0.001,
                 init_val: float=0.05):

        """
            Constructor for the class implementing a Bayesian neural network
            trained with the probabilistic back propagation method.

            @param X_train      Matrix with the features for the training data.
            @param y_train      Vector with the target variables for the
                                training data.
            @param n_hidden     Vector with the number of neurons for each
                                hidden layer.
            @param n_epochs     Numer of epochs for which to train the
                                network. The recommended value 40 should be
                                enough.
            @param normalize    Whether to normalize the input features. This
                                is recommended unles the input vector is for
                                example formed by binary features (a
                                fingerprint). In that case we do not recommend
                                to normalize the features.
            @param tau          Tau value used for regularization
            @param regularizer  l2: l2 regularization
                                gr: gaussian prior regularization
        """

        self.use_cov = use_cov
        self.per_activation_rate = per_activation_rate
        self._previous_noise_param = None
        self.init_val = init_val

        # We normalize the training data to have zero mean and unit standard
        # deviation in the training set if necessary

        super(net_learn_drop_rate, self).__init__(debug=debug)
        self.normalize_data(X_train, y_train, X_val, y_val)

        # regularizer
        N = self.X_train.shape[0]
        lengthscale = 1e-2
        self.reg_scale = lengthscale ** 2 / (2. * N * tau)
        self.regularizer = regularizer

        # We construct the network
        batch_size = 128
        inputs = Input(shape=(X_train.shape[1],))
        mean, var = self.get_dropout_layer_block(inputs, dropout_rate=dropout_rate)
        mean, var = self.get_dense_layer_block(mean, var, nh=n_hidden[0], activation='relu')

        for i in range(len(n_hidden) - 1):
            mean, var = self.get_dropout_layer_block(mean, var=var, dropout_rate=dropout_rate)
            mean, var = self.get_dense_layer_block(mean, var, nh=n_hidden[i + 1], activation='relu')

        mean, var = self.get_dropout_layer_block(mean, var=var, dropout_rate=dropout_rate)
        mean, var = self.get_dense_layer_block(mean, var, nh=self.y_train_normalized.shape[1])

        var = AssertionLayer(assert_non_negative, message='After last dense!')(var)

        # merge variance and mean
        if self.use_cov:
            cov_diag = Lambda(lambda x:  tf.linalg.diag_part(x))(var)
            cov_diag = AddHomoscedasticUncertaintyLayer()(cov_diag)
            merged_output = Lambda(lambda x: tf.concat([x[0], x[1]], axis=-1))([mean, cov_diag])
        else:
            merged_output = Lambda(lambda x: tf.concat([x[0], x[1]], axis=-1))([mean, var])

        model = Model(inputs, merged_output)

        opt = Adam(lr=lr)
        model.compile(loss=learn_dropout_loss, optimizer=opt)

        # add dummy targets
        self.y_train_normalized = np.concatenate([self.y_train_normalized, np.ones(self.y_train_normalized.shape)], axis=-1)

        # We iterate the learning process
        start_time = time.time()
        model.fit(self.X_train, self.y_train_normalized, validation_data=(self.X_val, self.y_val_normalized),
                  batch_size=batch_size, epochs=n_epochs, verbose=0, callbacks=self.get_callbacks())
        self.model = model
        self.tau = tau
        self.dropout = dropout_rate
        self.running_time = time.time() - start_time

        # We are done!

    def predict(self, X_test, y_test,
                analytic_log_likelihood: bool = False):

        """
            Function for making predictions with the Bayesian neural network.

            @param X_test   The matrix of features for the test data
            @analytic_log_likelihood    compute TLL analytically for variance propagation
                                        under gaussian assumption

            @return m       The predictive mean for the test target variables.
            @return v       The predictive variance for the test target
                            variables.
            @return v_noise The estimated variance for the additive noise.

        """

        X_test = np.array(X_test, ndmin=2)
        y_test = np.array(y_test, ndmin=2).T

        # We normalize the test set

        X_test = (X_test - np.full(X_test.shape, self.mean_X_train)) / \
                 np.full(X_test.shape, self.std_X_train)

        # We compute the predictive mean and variance for the target variables
        # of the test data

        model = self.model

        start = time.time()
        pred = model.predict(X_test, batch_size=500, verbose=1)
        runtime = time.time() - start
        print('Runtime LDO:', runtime)

        # get mean and std
        mean = pred[:, :1]
        std = np.sqrt(pred[:, 1:])

        # two ways of determining the test log-likelihood
        if self.tau is None:
            mean = mean * self.std_y_train + self.mean_y_train
            ll = -0.5 * (y_test[None] - mean) ** 2 / std - 0.5 * np.log(2 * np.pi) - 0.5 * np.log(std)
            test_ll = np.mean(ll)
        else:
            # sample from distribution to compute test log-likelihood
            samples_norm = np.array([np.random.normal(loc=mean, scale=std) for _ in range(T)])
            mean = mean * self.std_y_train + self.mean_y_train
            samples = samples_norm * self.std_y_train + self.mean_y_train

            ll = (logsumexp(-0.5 * self.tau * (y_test[None] - samples) ** 2., 0) - np.log(T)
                  - 0.5 * np.log(2 * np.pi) + 0.5 * np.log(self.tau))
            test_ll = np.mean(ll)

        rmse_pred = np.mean((y_test.squeeze() - mean.squeeze()) ** 2.) ** 0.5

        # We are done!
        return rmse_pred, test_ll, runtime

    def get_dense_layer_block(self, mean, var, nh: int=10, activation: str=None):
        dense_layer =  Dense(nh, activation=activation,
                             kernel_regularizer=self.get_regularizer())
        mean = dense_layer(mean)
        var = DenseVarPropagationLayer(dense_layer, use_cov=self.use_cov)(var)

        if activation is not None and activation=='relu':
            pre_activation_values = PreActivationLayer(dense_layer)(dense_layer.input)
            var = ReLUActivationVarPropagationLayer(pre_activation_values, use_cov=self.use_cov)(var)

        return mean, var

    def get_dropout_layer_block(self, mean, var=None, dropout_rate: float=0.5):
        dropout_layer = Dropout(dropout_rate)
        # mean = dropout_layer(mean, training=False)
        if var is not None:
            noise_layer = LearnDropoutVarPropagationLayer(dropout_layer,
                                                          per_activation_rate=self.per_activation_rate,
                                                          use_cov=self.use_cov,
                                                          initial_noise=False)
            mean, var = noise_layer([mean, var])
        else:
            noise_layer = LearnDropoutVarPropagationLayer(dropout_layer,
                                                          per_activation_rate=self.per_activation_rate,
                                                          use_cov=self.use_cov,
                                                          initial_noise=True)
            mean, var = noise_layer(mean)

        self._previous_noise_param = 1 - tf.nn.sigmoid(noise_layer.rate)
        return mean, var

    def get_regularizer(self):
        if self.regularizer == 'l2':
            return l2(l=self.reg_scale)
        elif self.regularizer == 'gr':
            if self._previous_noise_param is None:
                assert False, 'Need preceeding noise layer!'
            return gpreg(self._previous_noise_param)
        elif self.regularizer is None:
            return l2(l=0.0)
        else:
            raise NotImplementedError
