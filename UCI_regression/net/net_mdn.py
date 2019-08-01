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
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

from net.losses import mdn_loss_wrapper
from net.base_net import BaseNet
from net.layers import AddHomoscedasticUncertaintyLayer


MODES = 1
T = 10000


class net_mdn(BaseNet):

    def __init__(self, X_train, y_train, X_val, y_val, n_hidden, n_epochs=40,
                 tau: int=None, debug: bool=False, lr: float=0.001):
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
        """

        # We normalize the training data to have zero mean and unit standard
        # deviation in the training set if necessary

        super(net_mdn, self).__init__(debug=debug)
        self.normalize_data(X_train, y_train, X_val, y_val)

        # We construct the network
        N = self.X_train.shape[0]
        batch_size = 128
        lengthscale = 1e-2
        reg = lengthscale ** 2 / (2. * N * tau)
        inputs = Input(shape=(X_train.shape[1],))
        inter = Dense(n_hidden[0], activation='relu', kernel_regularizer=l2(reg))(inputs)

        for i in range(len(n_hidden) - 1):
            inter = Dense(n_hidden[i + 1], activation='relu', kernel_regularizer=l2(reg))(inter)

        # outputs = Dense(y_train_normalized.shape[1]*3*MODES, kernel_regularizer=l2(0.01))(inter)
        outputs_mean = Dense(self.y_train_normalized.shape[1] * MODES, kernel_regularizer=l2(reg))(inter)
        outputs_var = Dense(self.y_train_normalized.shape[1] * MODES, kernel_regularizer=l2(reg), activation=nnelu)(inter)
        outputs_var = AddHomoscedasticUncertaintyLayer()(outputs_var)
        outputs = Lambda(lambda x: tf.concat(x, axis=-1))([outputs_mean, outputs_var])

        model = Model(inputs, outputs)

        opt = Adam(lr=lr)
        model.compile(loss=mdn_loss_wrapper(modes=MODES), optimizer=opt)

        # add dummy targets
        y_train_normalized = self.add_dummy_labels(self.y_train_normalized)

        # We iterate the learning process
        start_time = time.time()
        model.fit(self.X_train, self.y_train_normalized, validation_data=(self.X_val, self.y_val_normalized),
                  batch_size=batch_size, epochs=n_epochs, verbose=0, callbacks=self.get_callbacks())
        self.model = model
        self.tau = tau
        self.running_time = time.time() - start_time

        # We are done!

    def predict(self, X_test, y_test):

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

        start = time.time()
        pred = self.model.predict(X_test, batch_size=500, verbose=1)
        runtime = time.time() - start
        print('Runtime MDN:', runtime)
        mean, std = self.extract_mean_std(pred)

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

    def add_dummy_labels(self, y):
        dummy_labels = np.ones((y.shape[0], 3*MODES-1))
        return np.concatenate((y, dummy_labels), axis=1)

    def extract_mean_std(self, prediction):
        mean = prediction[:, :MODES]
        std = np.sqrt(prediction[:, MODES:2*MODES])
        return mean, std


def nnelu(inputs):
    """
    Computes the Non-Negative Exponential Linear Unit
    """
    return tf.add(tf.constant(1, dtype=tf.float32), tf.nn.elu(inputs))
