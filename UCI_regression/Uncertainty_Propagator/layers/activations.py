import tensorflow as tf

from ..utils.helper import relu_var_tf, d_softmax_tf, jacobian_tanh_tf, jacobian_ReLU_tf
from ..utils.low_rank_approximation_utils import mult_lr_cov
from ..layers.layers import ActivationVarPropagationLayer


class LinearActivationVarPropagationLayer(ActivationVarPropagationLayer):
    """
    Linear activation function. Identity...
    """

    def __init__(self, inputs, layer=None, use_cov=False, **kwargs):
        super(LinearActivationVarPropagationLayer, self).__init__(inputs, layer=layer, use_cov=use_cov, **kwargs)

    def call(self, x):
        return x


class ReLUActivationVarPropagationLayer(ActivationVarPropagationLayer):
    """
    Variance propagation through ReLU activation function.
    """

    def __init__(self, inputs, layer=None, use_cov=False, **kwargs):
        super(ReLUActivationVarPropagationLayer, self).__init__(inputs, layer=layer, use_cov=use_cov, **kwargs)
        
    def _call_full_cov_approx(self, x):
        """
        approximate propagation of full covariance matrix
        """
        return tf.multiply(x, tf.einsum('ij,ik->ijk', tf.to_float(self.inputs > 0), tf.to_float(self.inputs > 0)))
    
    def _call_diag_cov_exact(self, x):
        """
        approximate propagation with diagonal covariance matrix
        """
        return relu_var_tf(self.inputs, x)
    
    def _call_diag_cov_approx(self, x):
        """
        approximate propagation with diagonal covariance matrix and exact variance computation
        under gaussian assumption
        """
        return tf.multiply(x, tf.to_float(self.inputs > 0))

    def _call_low_rank_cov(self, x):
        J = jacobian_ReLU_tf(self.inputs)
        return mult_lr_cov(x, J)


class TanhActivationVarPropagationLayer(ActivationVarPropagationLayer):
    """
    Variance propagation through ReLU activation function.
    """

    def __init__(self, inputs, layer=None, use_cov=False, **kwargs):
        super(TanhActivationVarPropagationLayer, self).__init__(inputs, layer=layer, use_cov=use_cov, **kwargs)
    
    def _call_diag_cov_approx(self, x):
        """
        approximate propagation with diagonal covariance matrix and exact variance computation
        under gaussian assumption
        """
        J = jacobian_tanh_tf(self.inputs)
        return tf.multiply(x, J**2)


class SoftmaxActivationVarPropagationLayer(ActivationVarPropagationLayer):

    def __init__(self, inputs, layer=None, use_cov=False, **kwargs):
        if 'soft_exact' in kwargs:
            kwargs['exact'] = kwargs['soft_exact']
            del kwargs['soft_exact']
        else:
            kwargs['exact'] = False
        super(SoftmaxActivationVarPropagationLayer, self).__init__(inputs, layer=layer, use_cov=use_cov, **kwargs)
    
    def _call_diag_cov_approx(self, x):
        """
        approximate propagation with diagonal covariance matrix
        """
        softmax = tf.nn.softmax(self.inputs, axis=-1)
        return d_softmax_tf(x, softmax)
