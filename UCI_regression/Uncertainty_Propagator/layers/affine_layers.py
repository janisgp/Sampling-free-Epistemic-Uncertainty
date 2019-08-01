import tensorflow as tf
from tensorflow.keras.layers import Layer

from ..layers.layers import VarPropagationLayer
from ..utils.low_rank_approximation_utils import mult_lr_cov


class DenseVarPropagationLayer(VarPropagationLayer):

    def __init__(self, dense_layer, use_cov=False, **kwargs):
        self.kernel_weights = dense_layer.weights[0]
        super(DenseVarPropagationLayer, self).__init__(dense_layer, use_cov, **kwargs)

    def _call_diag_cov(self, x):
        return tf.tensordot(x, self.kernel_weights**2, axes=[[1], [0]])

    def _call_full_cov(self, x):
        return tf.tensordot(tf.tensordot(x, self.kernel_weights, [[2], [0]]), 
                               self.kernel_weights, [[1], [0]])

    def _call_low_rank_cov(self, x):
        return mult_lr_cov(x, self.kernel_weights)


class Conv2DVarPropagationLayer(VarPropagationLayer):

    def __init__(self, conv2d_layer, use_cov=False, **kwargs):
        super(Conv2DVarPropagationLayer, self).__init__(conv2d_layer, use_cov=use_cov, **kwargs)
    
    def _call_diag_cov(self, x):
        return tf.nn.convolution(x, 
                                 self.layer.kernel**2,
                                 self.layer.padding.upper(),
                                 strides=self.layer.strides)


class Conv2DTransposeVarPropagationLayer(VarPropagationLayer):

    def __init__(self, conv2d_layer, use_cov=False, **kwargs):
        super(Conv2DTransposeVarPropagationLayer, self).__init__(conv2d_layer, use_cov=use_cov, **kwargs)
        self.strides = tuple([1] + list(self.layer.strides) + [1])

    def _call_diag_cov(self, x):
        return tf.nn.conv2d_transpose(x,
                                      self.layer.kernel ** 2,
                                      tf.shape(self.layer.output),
                                      self.strides,
                                      padding=self.layer.padding.upper())


class PreActivationLayer(Layer):

    def __init__(self, layer, **kwargs):
        self.params = layer.weights
        self.layer_type = layer.__class__.__name__
        self.config = layer.get_config()
        config = layer.get_config()
        self.layer = layer
        super(PreActivationLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(PreActivationLayer, self).build(input_shape)

    def call(self, x):
        if self.layer_type == 'Dense':
            out = self._call_dense(x)
        elif self.layer_type == 'Conv2D':
            out = self._call_conv2d(x)
        return out
    
    def _call_dense(self, x):
        return tf.tensordot(x, self.params[0], axes=1) + self.params[1]
        
    def _call_conv2d(self, x):
        return tf.nn.convolution(x,self.params[0],self.config['padding'].upper(),strides=self.config['strides'])+self.params[1]

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)
