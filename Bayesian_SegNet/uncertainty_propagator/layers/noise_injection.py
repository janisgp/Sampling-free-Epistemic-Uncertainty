import tensorflow as tf
from keras.layers import Lambda
from ..layers.affine_layers import VarPropagationLayer
from ..utils.helper import variance_product_rnd_vars


class DropoutVarPropagationLayer(VarPropagationLayer):

    def __init__(self, noise_layer, initial_noise=False, **kwargs):
        self.initial_noise = initial_noise
        self.rate = noise_layer.rate
        super(DropoutVarPropagationLayer, self).__init__(noise_layer, **kwargs)

    def _call_diag_cov(self, x):
        if self.initial_noise:
            out = x**2*self.rate/(1-self.rate)
        else:
            new_mean = 1-self.rate
            new_var = self.rate*(1-self.rate)
            mean = self.layer.input
            var = x
            out = variance_product_rnd_vars(mean, new_mean, var, new_var)/(1-self.rate)**2
        return out
    
    def _call_full_cov(self, x):
        if self.initial_noise:
            out = x**2*self.rate/(1-self.rate)
            out = tf.linalg.diag(out)
        else:
            new_mean = 1-self.rate
            new_var = self.rate*(1-self.rate)
            mean = self.layer.input/(1-self.rate)
            var = tf.matrix_diag_part(x)
            prod_var = variance_product_rnd_vars(mean, new_mean, var, new_var)/(1-self.rate)**2
            out = x - tf.matrix_diag(var) + tf.matrix_diag(prod_var)
        return out
