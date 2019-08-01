import numpy as np
import tensorflow as tf


def inverse_sigmoid_np(p):
    return -1*np.log(1/p-1)


def sigmoid_np(x):
    return 1/(1+np.exp(-1*x))


def print_noise_layer_weights(model: tf.keras.models.Model):
    i = 1
    for l in model.layers:
        class_name = l.__class__.__name__
        if class_name == 'LearnGaussianNoiseVarPropagationLayer':
            rate, mean = l.get_weights()
            print(str(i) + '. noise layer: LearnGaussianNoiseVarPropagationLayer')
            print('Mean:')
            print(mean)
            print('Standard deviation:')
            print(np.exp(rate))
        elif class_name == 'LearnDropoutVarPropagationLayer':
            rate = l.get_weights()
            print(str(i) + '. noise layer: LearnDropoutVarPropagationLayer')
            print('Dropout rate:')
            print(sigmoid_np(np.array(rate)))
        i += 1
