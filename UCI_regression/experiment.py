# Copyright 2019, Janis Postels, All rights reserved.
# This code is based on the code by Yarin Gal used for his
# paper "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning".

import math
import numpy as np
import argparse
import sys
import tensorflow.keras.backend as K
import os

from net.net import net
from net.net_mdn import net_mdn
from net.net_learn_drop_rate import net_learn_drop_rate

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser=argparse.ArgumentParser()

parser.add_argument('--dir', '-d', required=True, help='Name of the UCI Dataset directory. Eg: bostonHousing')
parser.add_argument('--epochx', '-e', default=500, type=int, help='Multiplier for the number of epochs for training.')
parser.add_argument('--hidden', '-nh', default=2, type=int, help='Number of hidden layers for the neural net')
parser.add_argument('--full_cov', '-fc', default=0, type=int, help='Use full covariance for uncertainty propagation')
parser.add_argument('--full_cov_drop', '-fcd', default=0, type=int, help='Use full covariance for dropout rate learning')
parser.add_argument('--per_activation_rate', '-par', default=0, type=int, help='Use one dropout rate per activation')
parser.add_argument('--gpu', '-g', default='0', type=str, help='GPU used during training')
parser.add_argument('--dropout', '-do', default='0', type=int, help='Use monte-carlo dropout')
parser.add_argument('--mdn', '-m', default='0', type=int, help='Use mixture density network')
parser.add_argument('--learn_dropout', '-ld', default='0', type=int, help='Learn Dropout Rate')
parser.add_argument('--debug', '-db', default='0', type=int, help='Debug mode')
parser.add_argument('--learning_rate', '-lr', default='0.001', type=float, help='Learning rate')
parser.add_argument('--initial_drop_rate', '-id', default='0.05', type=float, help='Initial dropout rate')

args=parser.parse_args()

data_directory = args.dir
epochs_multiplier = args.epochx
num_hidden_layers = args.hidden
use_cov = args.full_cov == 1
use_cov_drop = args.full_cov_drop == 1
per_activation_rate = args.per_activation_rate == 1
exact = args.exact_act == 1
mdn = args.mdn == 1
use_mc_dropout = args.dropout == 1
analytic_tll = args.analytic_tll == 1
learn_dropout = args.learn_dropout == 1
gpu = args.gpu
debug = args.debug == 1
lr = args.learning_rate
initial_dropout_rate = args.initial_drop_rate

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
set_session(tf.Session(config=config))

sys.path.append('net/')

# We delete previous results

from subprocess import call

data_path = "./UCI_Datasets/" + data_directory
results_dir = '/results/'
if not os.path.exists(data_path + '/results/analytic'):
    os.makedirs(data_path + '/results/analytic')
if not os.path.exists(data_path + '/results/mdn'):
    os.makedirs(data_path + '/results/mdn')
if not os.path.exists(data_path + '/results/ldo'):
    os.makedirs(data_path + '/results/ldo')
if not os.path.exists(data_path + '/results/mc_dropout'):
    os.makedirs(data_path + '/results/mc_dropout')
results_dir_ana = results_dir + 'analytic/'
results_dir_mdn = results_dir + 'mdn/'
results_dir_ldo = results_dir + 'ldo/'
results_dir_mc_dropout = results_dir + 'mc_dropout/'

# paths mc dropout
_RESULTS_TEST_LOG = data_path + results_dir + "log_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
# paths approximate dropout
_RESULTS_TEST_LOG_ANA = data_path + results_dir_ana + "log_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
# paths mdn
_RESULTS_TEST_LOG_MDN = data_path + results_dir_mdn + "log_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
# paths learned dropout rate
_RESULTS_TEST_LOG_LDO = data_path + results_dir_ldo + "log_" + str(epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"

_DATA_DIRECTORY_PATH = data_path + "/data/"
_DROPOUT_RATES_FILE = _DATA_DIRECTORY_PATH + "dropout_rates.txt"
_TAU_VALUES_FILE = _DATA_DIRECTORY_PATH + "tau_values.txt"
_DATA_FILE = _DATA_DIRECTORY_PATH + "data.txt"
_HIDDEN_UNITS_FILE = _DATA_DIRECTORY_PATH + "n_hidden.txt"
_EPOCHS_FILE = _DATA_DIRECTORY_PATH + "n_epochs.txt"
_INDEX_FEATURES_FILE = _DATA_DIRECTORY_PATH + "index_features.txt"
_INDEX_TARGET_FILE = _DATA_DIRECTORY_PATH + "index_target.txt"
_N_SPLITS_FILE = _DATA_DIRECTORY_PATH + "n_splits.txt"

def _get_index_train_test_path(split_num, train = True):
    """
       Method to generate the path containing the training/test split for the given
       split number (generally from 1 to 20).
       @param split_num      Split number for which the data has to be generated
       @param train          Is true if the data is training data. Else false.
       @return path          Path of the file containing the requried data
    """
    if train:
        return _DATA_DIRECTORY_PATH + "index_train_" + str(split_num) + ".txt"
    else:
        return _DATA_DIRECTORY_PATH + "index_test_" + str(split_num) + ".txt" 


print ("Removing existing result files...")
if use_mc_dropout:
    call(["rm", _RESULTS_TEST_LOG])
    call(["rm", _RESULTS_TEST_LOG_ANA])
if mdn:
    call(["rm", _RESULTS_TEST_LOG_MDN])
if learn_dropout:
    call(["rm", _RESULTS_TEST_LOG_LDO])
print ("Result files removed.")

# We fix the random seed

np.random.seed(1)

print ("Loading data and other hyperparameters...")
# We load the data

data = np.loadtxt(_DATA_FILE)

# We load the number of hidden units

n_hidden = np.loadtxt(_HIDDEN_UNITS_FILE).tolist()

# We load the number of training epocs

n_epochs = np.loadtxt(_EPOCHS_FILE).tolist()

# We load the indexes for the features and for the target

index_features = np.loadtxt(_INDEX_FEATURES_FILE)
index_target = np.loadtxt(_INDEX_TARGET_FILE)

X = data[ : , [int(i) for i in index_features.tolist()] ]
y = data[ : , int(index_target.tolist()) ]

# We iterate over the training test splits

n_splits = np.loadtxt(_N_SPLITS_FILE)
print ("Done.")

errors, MC_errors, MC_lls, ANA_errors, ANA_lls, mc_times, ana_times = [], [], [], [], [], [], []
MDN_errors, MDN_lls, mdn_times = [], [], []
LDO_errors, LDO_lls, ldo_times = [], [], []
for split in range(int(n_splits)):

    # We load the indexes of the training and test sets
    print ('Loading file: ' + _get_index_train_test_path(split, train=True))
    print ('Loading file: ' + _get_index_train_test_path(split, train=False))
    index_train = np.loadtxt(_get_index_train_test_path(split, train=True))
    index_test = np.loadtxt(_get_index_train_test_path(split, train=False))

    X_train = X[ [int(i) for i in index_train.tolist()] ]
    y_train = y[ [int(i) for i in index_train.tolist()] ]
    
    X_test = X[ [int(i) for i in index_test.tolist()] ]
    y_test = y[ [int(i) for i in index_test.tolist()] ]

    X_train_original = X_train
    y_train_original = y_train
    num_training_examples = int(0.8 * X_train.shape[0])
    X_validation = X_train[num_training_examples:, :]
    y_validation = y_train[num_training_examples:]
    X_train = X_train[0:num_training_examples, :]
    y_train = y_train[0:num_training_examples]
    
    # Printing the size of the training, validation and test sets
    print ('Number of training examples: ' + str(X_train.shape[0]))
    print ('Number of validation examples: ' + str(X_validation.shape[0]))
    print ('Number of test examples: ' + str(X_test.shape[0]))
    print ('Number of train_original examples: ' + str(X_train_original.shape[0]))

    # List of hyperparameters which we will try out using grid-search
    dropout_rates = np.loadtxt(_DROPOUT_RATES_FILE).tolist()
    tau_values = np.loadtxt(_TAU_VALUES_FILE).tolist()

    if use_mc_dropout:

        # We perform grid-search to select the best hyperparameters based on the highest log-likelihood value
        best_network = None
        best_ll_MC = -float('inf')
        best_tau_MC = 0
        best_dropout_MC = 0
        best_ll_ANA = -float('inf')
        best_tau_ANA = 0
        best_dropout_ANA = 0
        for dropout_rate in dropout_rates:
            for tau in tau_values:
                print ('Grid search step: Tau: ' + str(tau) + ' Dropout rate: ' + str(dropout_rate))
                network = net(X_train, y_train, X_validation, y_validation,
                              ([ int(n_hidden) ] * num_hidden_layers),
                              n_epochs = int(n_epochs * epochs_multiplier), tau = tau,
                              dropout = dropout_rate, debug=debug, lr=lr)

                # We obtain the test RMSE and the test ll from the validation sets

                error, MC_error, MC_ll, ANA_error, ANA_ll, mc_runtime, ana_runtime = network.predict(X_validation,
                                                                                                     y_validation,
                                                                                                     use_cov=use_cov)

                mc_times.append(mc_runtime)
                ana_times.append(ana_runtime)

                if (MC_ll > best_ll_MC):
                    best_ll_MC = MC_ll
                    best_tau_MC = tau
                    best_dropout_MC = dropout_rate
                    print ('Best MC log_likelihood changed to: ' + str(best_ll_MC))
                    print ('Best MC tau changed to: ' + str(best_tau_MC))
                    print ('Best MC dropout rate changed to: ' + str(best_dropout_MC))
                if (ANA_ll > best_ll_ANA):
                    best_ll_ANA = ANA_ll
                    best_tau_ANA = tau
                    best_dropout_ANA = dropout_rate
                    print ('Best ANA log_likelihood changed to: ' + str(best_ll_ANA))
                    print ('Best ANA tau changed to: ' + str(best_tau_ANA))
                    print ('Best ANA dropout rate changed to: ' + str(best_dropout_ANA))

                # remove stale GPU memory
                K.clear_session()
                del network


        # Storing test results
        best_network_MC = net(X_train_original, y_train_original, X_test, y_test,
                              ([ int(n_hidden) ] * num_hidden_layers),
                              n_epochs = int(n_epochs * epochs_multiplier), tau = best_tau_MC,
                              dropout = best_dropout_MC, debug=debug, lr=lr)
        error, MC_error, MC_ll, ANA_error, ANA_ll, _, _ = best_network_MC.predict(X_test, y_test, use_cov=use_cov)

        errors += [error]
        MC_errors += [MC_error]
        MC_lls += [MC_ll]

        # remove stale GPU memory
        K.clear_session()
        del best_network_MC

        # Storing test results
        best_network_ANA = net(X_train_original, y_train_original, X_test, y_test,
                              ([ int(n_hidden) ] * num_hidden_layers),
                              n_epochs = int(n_epochs * epochs_multiplier), tau = best_tau_ANA,
                              dropout = best_dropout_ANA, debug=debug, lr=lr)
        error, MC_error, MC_ll, ANA_error, ANA_ll, _, _ = best_network_ANA.predict(X_test, y_test, use_cov=use_cov)

        ANA_errors += [ANA_error]
        ANA_lls += [ANA_ll]

        # remove stale GPU memory
        K.clear_session()
        del best_network_ANA

    if mdn:

        best_ll_MDN = -float('inf')
        best_tau_MDN = 0
        for tau in tau_values:

            network_mdn = net_mdn(X_train, y_train, X_validation, y_validation,
                                  ([int(n_hidden)] * num_hidden_layers),
                                  n_epochs=int(n_epochs * epochs_multiplier),
                                  tau=tau, debug=debug, lr=lr)

            MDN_error, MDN_ll, MDN_runtime = network_mdn.predict(X_validation, y_validation)

            if MDN_ll > best_ll_MDN:
                best_tau_MDN = tau
                best_ll_MDN = MDN_ll
                print('Best MDN log_likelihood changed to: ' + str(best_ll_MDN))
                print('Best MDN tau changed to: ' + str(best_tau_MDN))

            # remove stale GPU memory
            K.clear_session()
            del network_mdn

        best_mdn = net_mdn(X_train_original, y_train_original, X_test, y_test, ([int(n_hidden)] * num_hidden_layers),
                           n_epochs=int(n_epochs * epochs_multiplier), tau=best_tau_MDN, debug=debug, lr=lr)

        MDN_error, MDN_ll, MDN_runtime = best_mdn.predict(X_test, y_test)

        MDN_errors += [MDN_error]
        MDN_lls += [MDN_ll]
        mdn_times += [MDN_runtime]

        # remove stale GPU memory
        K.clear_session()
        del best_mdn

    if learn_dropout:

        best_ll_LDO = -float('inf')
        best_tau_LDO = 0
        for tau in tau_values:

            network_ldo = net_learn_drop_rate(X_train, y_train,
                                              X_validation, y_validation,
                                              ([int(n_hidden)] * num_hidden_layers),
                                              n_epochs=int(n_epochs * epochs_multiplier),
                                              tau=tau,
                                              dropout_rate=0.5,
                                              per_activation_rate=per_activation_rate,
                                              use_cov=use_cov_drop,
                                              regularizer='gr',
                                              debug=debug,
                                              lr=lr,
                                              init_val=initial_dropout_rate)

            LDO_error, LDO_ll, LDO_runtime = network_ldo.predict(X_validation, y_validation)

            if LDO_ll > best_ll_LDO:
                best_tau_LDO = tau
                best_ll_LDO = LDO_ll
                print('Best LDO log_likelihood changed to: ' + str(best_ll_LDO))
                print('Best LDO tau changed to: ' + str(best_tau_LDO))

            # remove stale GPU memory
            K.clear_session()
            del network_ldo

        best_ldo = net_learn_drop_rate(X_train_original, y_train_original,
                                       X_test, y_test,
                                       ([int(n_hidden)] * num_hidden_layers),
                                       n_epochs=int(n_epochs * epochs_multiplier),
                                       dropout_rate=0.5,
                                       tau=best_tau_LDO,
                                       per_activation_rate=per_activation_rate,
                                       use_cov=use_cov_drop,
                                       regularizer='gr',
                                       debug=debug,
                                       lr=lr,
                                       init_val=initial_dropout_rate)

        LDO_error, LDO_ll, LDO_runtime = best_ldo.predict(X_test, y_test)

        LDO_errors += [LDO_error]
        LDO_lls += [LDO_ll]
        ldo_times += [LDO_runtime]

        # remove stale GPU memory
        K.clear_session()
        del best_ldo

    print ("Tests on split " + str(split) + " complete.")

if use_mc_dropout:
    with open(_RESULTS_TEST_LOG, "a") as myfile:
        myfile.write('errors %f +- %f (stddev) +- %f (std error), median %f 25p %f 75p %f \n' % (
            np.mean(errors), np.std(errors), np.std(errors)/math.sqrt(n_splits),
            np.percentile(errors, 50), np.percentile(errors, 25), np.percentile(errors, 75)))
        myfile.write('MC errors %f +- %f (stddev) +- %f (std error), median %f 25p %f 75p %f \n' % (
            np.mean(MC_errors), np.std(MC_errors), np.std(MC_errors)/math.sqrt(n_splits),
            np.percentile(MC_errors, 50), np.percentile(MC_errors, 25), np.percentile(MC_errors, 75)))
        myfile.write('lls %f +- %f (stddev) +- %f (std error), median %f 25p %f 75p %f \n' % (
            np.mean(MC_lls), np.std(MC_lls), np.std(MC_lls)/math.sqrt(n_splits),
            np.percentile(MC_lls, 50), np.percentile(MC_lls, 25), np.percentile(MC_lls, 75)))
        myfile.write('MC dropout average runtime for test set prediction: %f' % (np.mean(mc_times)))

    with open(_RESULTS_TEST_LOG_ANA, "a") as myfile:
        myfile.write('MC errors %f +- %f (stddev) +- %f (std error), median %f 25p %f 75p %f \n' % (
            np.mean(ANA_errors), np.std(ANA_errors), np.std(ANA_errors)/math.sqrt(n_splits),
            np.percentile(ANA_errors, 50), np.percentile(ANA_errors, 25), np.percentile(ANA_errors, 75)))
        myfile.write('lls %f +- %f (stddev) +- %f (std error), median %f 25p %f 75p %f \n' % (
            np.mean(ANA_lls), np.std(ANA_lls), np.std(ANA_lls)/math.sqrt(n_splits),
            np.percentile(ANA_lls, 50), np.percentile(ANA_lls, 25), np.percentile(ANA_lls, 75)))
        myfile.write('Approximation dropout average runtime for test set prediction: %f' % (np.mean(ana_times)))

if mdn:
    with open(_RESULTS_TEST_LOG_MDN, "a") as myfile:
        myfile.write('MC errors %f +- %f (stddev) +- %f (std error), median %f 25p %f 75p %f \n' % (
            np.mean(MDN_errors), np.std(MDN_errors), np.std(MDN_errors)/math.sqrt(n_splits),
            np.percentile(MDN_errors, 50), np.percentile(MDN_errors, 25), np.percentile(MDN_errors, 75)))
        myfile.write('lls %f +- %f (stddev) +- %f (std error), median %f 25p %f 75p %f \n' % (
            np.mean(MDN_lls), np.std(MDN_lls), np.std(MDN_lls)/math.sqrt(n_splits),
            np.percentile(MDN_lls, 50), np.percentile(MDN_lls, 25), np.percentile(MDN_lls, 75)))
        myfile.write('MDN average runtime for test set prediction: %f' % (np.mean(mdn_times)))

if learn_dropout:
    with open(_RESULTS_TEST_LOG_LDO, "a") as myfile:
        myfile.write('MC errors %f +- %f (stddev) +- %f (std error), median %f 25p %f 75p %f \n' % (
            np.mean(LDO_errors), np.std(LDO_errors), np.std(LDO_errors)/math.sqrt(n_splits),
            np.percentile(LDO_errors, 50), np.percentile(LDO_errors, 25), np.percentile(LDO_errors, 75)))
        myfile.write('lls %f +- %f (stddev) +- %f (std error), median %f 25p %f 75p %f \n' % (
            np.mean(LDO_lls), np.std(LDO_lls), np.std(LDO_lls)/math.sqrt(n_splits),
            np.percentile(LDO_lls, 50), np.percentile(LDO_lls, 25), np.percentile(LDO_lls, 75)))
        myfile.write('LDO average runtime for test set prediction: %f' % (np.mean(ldo_times)))
