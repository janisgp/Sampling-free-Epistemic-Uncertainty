import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Lambda

from .utils.helper import batch_std
from .layers.affine_layers import PreActivationLayer
from .layer_link import *


class UncertaintyPropagator():

    def __init__(self, model, mc_samples: int=10, classification: bool=True):
        self.model = model
        self.mc_samples = mc_samples
        self.classification = classification
        self._mc_mode = True

    def build_model(self, use_cov=False, **kwargs):
        var = None
        for layer in self.model.layers:
            layer_type = layer.__class__.__name__
            if layer_type in noise_layers.keys():
                init = var is None
                var = noise_layers[layer_type](layer, initial_noise=init, use_cov=use_cov)(layer.input if init else var)
            elif var is not None:
                var = self._propagate_var(var, layer, use_cov, **kwargs)

        assert var is not None, 'Model has no noise layer!'

        std = Lambda(lambda x: tf.sqrt(x), name='compute_standard_deviation')(var)
        
        self.unc_model = Model(self.model.inputs, self.model.outputs + [std])

        return self.unc_model

    def _propagate_var(self, var, layer, use_cov, **kwargs):
        layer_type = layer.__class__.__name__
        if layer_type in pooling_layers.keys():
            return pooling_layers[layer_type](layer, use_cov=use_cov)(var)
        elif layer_type in affine_layers.keys():
            activation_name = layer.activation.__name__
            var = affine_layers[layer_type](layer, use_cov=use_cov)(var)
            if activation_name != 'linear':
                inputs = PreActivationLayer(layer)(layer.input)
            else:
                inputs = None
            return activation_layers[activation_name](inputs=inputs, use_cov=use_cov, **kwargs)(var)
        elif layer_type == 'Activation':
            activation_name = layer.activation.__name__
            return activation_layers[activation_name](inputs=layer.input, use_cov=use_cov, **kwargs)(var)
        else:
            print('Warning: Layer not implemented:', layer_type)
            if use_cov:
                raise NotImplementedError('Layer type not implemented!')
            return layer(var)
    
    def mc_prediction(self, X):
        f = K.function([self.model.layers[0].input, K.learning_phase()],
           [self.model.layers[-1].output])
        preds = np.array([f((X, 1))[0] for _ in range(self.mc_samples)])
        stds = preds.std(axis=0)
        if self.classification:
            stds = np.sqrt((stds**2).mean(-1))
        preds = preds.mean(axis=0)
        return preds, stds
        
    def approx_prediction(self, X):
        preds, stds = self.unc_model.predict(X)
        if self.classification:
            stds = np.sqrt((stds**2).mean(-1))
        return preds, stds
    
    def predict(self, X):
        if self._mc_mode:
            return self.mc_prediction(X)
        else:
            return self.approx_prediction(X)

    def set_mc_mode(self, mc_mode: bool):
        self._mc_mode = mc_mode