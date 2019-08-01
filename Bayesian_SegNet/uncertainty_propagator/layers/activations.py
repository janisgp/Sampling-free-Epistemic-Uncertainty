import math
import tensorflow as tf
from keras.layers import Lambda
from ..utils.helper import relu_var_tf, d_softmax_tf
from ..layers.layers import ActivationVarPropagationLayer


class LinearActivationVarPropagationLayer(ActivationVarPropagationLayer):

    def __init__(self, inputs, layer=None, use_cov=False, **kwargs):
        super(LinearActivationVarPropagationLayer, self).__init__(inputs, layer=layer, use_cov=use_cov, **kwargs)

    def call(self, x):
        return x
    
class ReLUActivationVarPropagationLayer(ActivationVarPropagationLayer):

    def __init__(self, inputs, layer=None, use_cov=False, **kwargs):
        super(ReLUActivationVarPropagationLayer, self).__init__(inputs, layer=layer, use_cov=use_cov, **kwargs)
        
    def _call_full_cov_approx(self, x):
        return tf.multiply(x, tf.einsum('ij,ik->ijk', tf.to_float(self.inputs > 0), tf.to_float(self.inputs > 0)))
    
    def _call_diag_cov_exact(self, x):
        return relu_var_tf(self.inputs, x)
    
    def _call_diag_cov_approx(self, x):
        return tf.multiply(x, tf.to_float(self.inputs > 0))

class SoftmaxActivationVarPropagationLayer(ActivationVarPropagationLayer):

    def __init__(self, inputs, layer=None, use_cov=False, **kwargs):
        if 'soft_exact' in kwargs:
            kwargs['exact'] = kwargs['soft_exact']
            del kwargs['soft_exact']
        else:
            kwargs['exact'] = False
        super(SoftmaxActivationVarPropagationLayer, self).__init__(inputs, layer=layer, use_cov=use_cov, **kwargs)
    
    def _call_diag_cov_approx(self, x):
        softmax = tf.nn.softmax(self.inputs, axis=-1)
        return d_softmax_tf(x, softmax)
