import tensorflow as tf
import tensorflow.keras.backend as K
from ..layers.layers import VarPropagationLayer


class BatchnormVarPropagationLayer(VarPropagationLayer):

    def __init__(self, norm_layer, use_cov=False, **kwargs):
        super(BatchnormVarPropagationLayer, self).__init__(norm_layer, use_cov, **kwargs)

    def _call_diag_cov(self, x, training=False):
        if training in {0, False}:
#             out = x * (self.layer.gamma / (self.layer.moving_variance + self.layer.epsilon))**2
            out = tf.nn.batch_normalization(x, 
                                           self.layer.moving_mean*0, 
                                           self.layer.moving_variance**2,
                                           self.layer.beta*0, 
                                           self.layer.gamma**2,
                                           self.layer.epsilon)
        else:
            x_shape = K.int_shape(x)
            reduction_axes = list(range(len(x_shape)))
            mean, var = tf.nn.moments(x, reduction_axes, None, None, False)
            out = x * (self.layer.gamma / (var + self.layer.epsilon))**2
        return out
