# Copyright 2016, Yarin Gal, All rights reserved.
# This code is based on the code by Jose Miguel Hernandez-Lobato used for his 
# paper "Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks".

import warnings
warnings.filterwarnings("ignore")

import time
from scipy.misc import logsumexp
import numpy as np

from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
import tensorflow.keras.backend as K

from Uncertainty_Propagator.uncertainty_propagator import UncertaintyPropagator
from net.base_net import BaseNet


class net(BaseNet):

    def __init__(self, X_train, y_train, X_val, y_val, n_hidden, n_epochs = 40,
                 tau = 1.0, dropout = 0.05, debug: bool=False, lr: float=0.001, 
                 normalize: bool=True):

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
            @param dropout      Dropout rate for all the dropout layers in the
                                network.
        """

        # We normalize the training data to have zero mean and unit standard
        # deviation in the training set if necessary

        super(net, self).__init__(debug=debug)
        if normalize:
            self.normalize_data(X_train, y_train, X_val, y_val)
        else:
            self.mean_X_train = np.zeros(X_train[0].shape)
            self.std_X_train = np.ones(X_train[0].shape)
            self.mean_y_train = np.zeros(y_train[0].shape)
            self.std_y_train = np.ones(y_train[0].shape)
            self.y_train_normalized = np.array(y_train, ndmin=2).T
            self.y_val_normalized = np.array(y_val, ndmin=2).T
            self.X_train = X_train
            self.X_val = X_val

        # We construct the network
        N = self.X_train.shape[0]
        batch_size = 128
        lengthscale = 1e-2
        reg = lengthscale**2 * (1 - dropout) / (2. * N * tau)
        inputs = Input(shape=(X_train.shape[1],))
        inter = Dropout(dropout)(inputs)
        inter = Dense(n_hidden[0], activation='relu', kernel_regularizer=l2(reg))(inter)
#         inter = Lambda(lambda x: tf.nn.relu(x))(inter)

        for i in range(len(n_hidden) - 1):
            inter = Dropout(dropout)(inter)
            inter = Dense(n_hidden[i+1], activation='relu', kernel_regularizer=l2(reg))(inter)

        inter = Dropout(dropout)(inter)
        outputs = Dense(self.y_train_normalized.shape[1], kernel_regularizer=l2(reg))(inter)
        model = Model(inputs, outputs)

        opt = Adam(lr=lr)
        model.compile(loss='mean_squared_error', optimizer=opt)

        # We iterate the learning process
        start_time = time.time()
        model.fit(self.X_train, self.y_train_normalized, validation_data=(self.X_val, self.y_val_normalized),
                  batch_size=batch_size, epochs=n_epochs, verbose=0, callbacks=self.get_callbacks())
        self.model = model
        self.tau = tau
        self.dropout = dropout
        self.running_time = time.time() - start_time

        # We are done!

    def predict(self, X_test, y_test, use_cov=False):

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

        X_test = np.array(X_test, ndmin = 2)
        y_test = np.array(y_test, ndmin = 2).T

        # We normalize the test set

        X_test = (X_test - np.full(X_test.shape, self.mean_X_train)) / \
            np.full(X_test.shape, self.std_X_train)

        # We compute the predictive mean and variance for the target variables
        # of the test data

        model = self.model
        standard_pred = model.predict(X_test, batch_size=500, verbose=1)
        standard_pred = standard_pred * self.std_y_train + self.mean_y_train
        rmse_standard_pred = np.mean((y_test.squeeze() - standard_pred.squeeze())**2.)**0.5
        
        T_mc = 100
        T_our = 10000
        
        f = K.function([model.layers[0].input, K.learning_phase()],
               [model.layers[-1].output])
        
        start = time.time()
        Yt_hat_norm = np.array([f((X_test, 1))[0] for _ in range(T_mc)])
        mc_runtime = time.time() - start
        print('Runtime Monte-Carlo: Dropout:', mc_runtime)
   
        Yt_hat = Yt_hat_norm * self.std_y_train + self.mean_y_train
        MC_pred = np.mean(Yt_hat, 0)
        rmse_mc = np.mean((y_test.squeeze() - MC_pred.squeeze())**2.)**0.5

        # We compute the test log-likelihood
        ll = (logsumexp(-0.5 * self.tau * (y_test[None] - Yt_hat)**2., 0) - np.log(T_mc)
            - 0.5*np.log(2*np.pi) + 0.5*np.log(self.tau))
        test_ll_mc = np.mean(ll)

        ### analytic computation of the predicted distribution ###

        # get model which also outputs the activations before dropout layer and deactivates dropout
        print('Get variance propagation model...')
        approximator = UncertaintyPropagator(model, use_cov=use_cov)
        ana_model = approximator.unc_model
        print('Loaded variance propagation model...')

        # predict
        start = time.time()
        pred, var = ana_model.predict(X_test, batch_size=500, verbose=1)
        ana_runtime = time.time() - start
        stds = np.sqrt(var)
        print('Runtime Approximation: Dropout:', ana_runtime)

        if use_cov:
            stds = stds[:, :, 0]

        # two ways of determining the test log-likelihood
        # sample from distribution to compute test log-likelihood
        samples_norm = np.array([np.random.normal(loc=pred, scale=stds) for i in range(T_our)])
        pred = pred * self.std_y_train + self.mean_y_train
        samples = samples_norm * self.std_y_train + self.mean_y_train

        ll = (logsumexp(-0.5 * self.tau * (y_test[None] - samples)**2., 0) - np.log(T_our)
            - 0.5*np.log(2*np.pi) + 0.5*np.log(self.tau))
        test_ll_ana = np.mean(ll)

        rmse_ana = np.mean((y_test.squeeze() - pred.squeeze()) ** 2.) ** 0.5
        print('Mean std difference:', np.abs(Yt_hat_norm.std(axis=0) - stds).mean() / Yt_hat_norm.std(axis=0).mean())

        # We are done!
        return rmse_standard_pred, rmse_mc, test_ll_mc, rmse_ana, test_ll_ana, mc_runtime, ana_runtime
