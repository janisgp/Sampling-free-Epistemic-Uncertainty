import tensorflow as tf

from ..layers.affine_layers import VarPropagationLayer
from ..utils.helper import variance_product_rnd_vars, covariance_elementwise_product_rnd_vec
from ..utils.low_rank_approximation_utils import get_cov_left_right


class DropoutVarPropagationLayer(VarPropagationLayer):

    def __init__(self, noise_layer: tf.keras.layers.Layer,
                 initial_noise: bool=False,
                 **kwargs):
        self.initial_noise = initial_noise
        self.rate = noise_layer.rate
        self.rate_tensor = tf.reshape(tf.constant(self.rate, dtype=tf.float32), [1])
        super(DropoutVarPropagationLayer, self).__init__(noise_layer, **kwargs)

    def _call_diag_cov(self, x):
        if self.initial_noise:
            out = x**2*self.rate/(1-self.rate)
        else:
            new_mean = 1-self.rate
            new_var = self.rate*(1-self.rate)
            mean = self.layer.input/self.rate
            var = x
            out = variance_product_rnd_vars(mean, new_mean, var, new_var)/(1-self.rate)**2
        return out

    def _call_full_cov(self, x):
        if self.initial_noise:
            out = x**2*self.rate/(1-self.rate)
            out = tf.linalg.diag(out)
        else:
            # new_mean = 1-self.rate
            # new_var = self.rate*(1-self.rate)
            # mean = self.layer.input/(1-self.rate)
            # var = tf.matrix_diag_part(x)
            # prod_var = variance_product_rnd_vars(mean, new_mean, var, new_var)/(1-self.rate)**2
            # out = x - tf.matrix_diag(var) + tf.matrix_diag(prod_var)

            mean = self.layer.input
            mean_shape = [mean.get_shape().as_list()[-1]]
            new_mean = tf.ones(mean_shape, dtype=tf.float32) * (1 - self.rate_tensor)
            new_var = tf.ones(mean_shape, dtype=tf.float32) * self.rate_tensor * (1 - self.rate_tensor)

            out = covariance_elementwise_product_rnd_vec(mean, x, new_mean, new_var)
        return out

    def _call_low_rank_cov(self, x):
        if self.initial_noise:
            out = x**2*self.rate/(1-self.rate)
            out = get_cov_left_right(out, self.low_rank_cov)
        else:
            # dummy, probably mathematically not feasible
            out = x
        return out
