import tensorflow as tf
from tensorflow.keras.initializers import Constant

from Uncertainty_Propagator.layers.noise_injection import DropoutVarPropagationLayer
from Uncertainty_Propagator.utils.helper import variance_product_rnd_vars, covariance_elementwise_product_rnd_vec
from net.utils import inverse_sigmoid_np


class LearnDropoutVarPropagationLayer(DropoutVarPropagationLayer):

    def __init__(self, noise_layer: tf.keras.layers.Layer,
                 per_activation_rate: bool=False,
                 init_val: float=0.05, **kwargs):
        self.per_activation_rate = per_activation_rate
        self.init_val = init_val
        super(LearnDropoutVarPropagationLayer, self).__init__(noise_layer, **kwargs)

    def build(self, input_shape):

        init_val = inverse_sigmoid_np(self.init_val)

        # determine weight shape
        if self.per_activation_rate:
            if isinstance(input_shape, tf.TensorShape):
                weight_shape = (input_shape[-1],)
            elif isinstance(input_shape, list):
                weight_shape = (input_shape[-1][-1],)
        else:
            weight_shape = (1,)

        # add weights to layer
        self.rate = self.add_weight(name=self.name + '_rate',
                                    shape=weight_shape,
                                    initializer=Constant(value=init_val),
                                    trainable=True)

        super(LearnDropoutVarPropagationLayer, self).build(input_shape)

    def _call_diag_cov(self, x):

        drop_rate = tf.nn.sigmoid(self.rate)

        if self.initial_noise:

            out_var = x**2 * drop_rate * (1 - drop_rate)
            out_mean = x * (1 - drop_rate)

        else:

            assert isinstance(x, list) and len(x) == 2

            new_mean = 1-drop_rate
            new_var = drop_rate*(1-drop_rate)
            out_var = variance_product_rnd_vars(x[0], new_mean, x[1], new_var)
            out_mean = x[0]*(1-drop_rate)

        return [out_mean, out_var]

    def _call_full_cov(self, x):

        drop_rate = tf.nn.sigmoid(self.rate)

        if self.initial_noise:

            out_var = x**2 * drop_rate * (1 - drop_rate)
            out_cov = tf.linalg.diag(out_var)
            out_mean = x * (1 - drop_rate)

        else:

            assert isinstance(x, list) and len(x) == 2

            out_cov = covariance_elementwise_product_rnd_vec(x[0], x[1], (1-drop_rate), drop_rate*(1-drop_rate))
            out_mean = x[0] * (1 - drop_rate)

        return [out_mean, out_cov]


class AddHomoscedasticUncertaintyLayer(tf.keras.layers.Layer):
    """
    Adding homoscedastic uncertainty variable
    """

    def build(self, input_shape):

        # add weights to layer
        self.log_variance = self.add_weight(name=self.name + '_rate',
                                            shape=(1,),
                                            initializer=Constant(value=-3),
                                            trainable=True)

        super(AddHomoscedasticUncertaintyLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs + tf.exp(self.log_variance)


class AssertionLayer(tf.keras.layers.Layer):
    """
    Layer for debugging with tf assertion
    """

    def __init__(self, assertion, message: str='', **kwargs):
        self.assertion = assertion
        self.message = message
        super(AssertionLayer, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        assert_op = self.assertion(inputs, message=self.message)
        with tf.control_dependencies([assert_op]):
            inputs = tf.debugging.check_numerics(inputs, self.message)
        return inputs


class AssertionLayerMainDiagonalBatch(AssertionLayer):
    """
    Layer for debugging with tf assertion limited to the
    main diagonal of a batch wise matrix
    """

    def call(self, inputs, **kwargs):
        var = tf.matrix_diag_part(inputs)
        var = super(AssertionLayerMainDiagonalBatch, self).call(var, **kwargs)
        diag = tf.matrix_diag(var)
        outputs = inputs - diag + diag
        return outputs
