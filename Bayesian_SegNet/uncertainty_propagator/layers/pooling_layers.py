import tensorflow as tf
import keras.backend as K
from keras.backend.tensorflow_backend import _preprocess_conv2d_input
from keras.backend.tensorflow_backend import _preprocess_padding
from ..layers.layers import VarPropagationLayer


class UpSampling2DVarPropagationLayer(VarPropagationLayer):

    def __init__(self, upsampling_layer, use_cov=False, **kwargs):
        super(UpSampling2DVarPropagationLayer, self).__init__(upsampling_layer, use_cov=False, **kwargs)
    
    def _call_diag_cov(self, x):
        return self.layer(x)
    
# class MemorizedUpSampling2DVarPropagationLayer(VarPropagationLayer):

#     def __init__(self, upsampling_layer, use_cov=False, **kwargs):
#         self.idx = upsampling_layer.idx
#         super(UpSampling2DVarPropagationLayer, self).__init__(upsampling_layer, use_cov=False, **kwargs)
    
#     def _call_diag_cov(self, x):
        
#         return _unpool2d_argmax(x, self.idx, self.layer.pool_size)
    
#     def _unpool2d_argmax(x: 'Tensor', idx: 'Tensor', pool_size: tuple) -> 'Tuple':
#         """
#         Un-pooling layer to complement pool2d_argmax.
#         Args:
#             x: input Tensor or variable
#             idx: index matching the shape of x from the pooling operation
#             pool_size: the pool_size used by the pooling operation
#         Returns:
#             an un-pooled version of x using indexes in idx
#         Notes:
#             follow the issue here for updates on this functionality:
#             https://github.com/tensorflow/tensorflow/issues/2169
#         """
#         # get the input shape of the tensor
#         ins = K.shape(x)
#         # create an index over the batches
#         batch_range = K.arange(K.cast(ins[0], 'int64'))
#         batch_range = K.reshape(batch_range, shape=[ins[0], 1, 1, 1])
#         # create a ones tensor in the shape of index
#         batch_idx = K.ones_like(idx) * batch_range
#         batch_idx = K.reshape(batch_idx, (-1, 1))
#         # create a complete index
#         index = K.reshape(idx, (-1, 1))
#         index = K.concatenate([batch_idx, index])

#         # get the output shape of the tensor
#         outs = [ins[0], ins[1] * pool_size[0], ins[2] * pool_size[1], ins[3]]
#         flat_output_shape = [outs[0], outs[1] * outs[2] * outs[3]]
#         # flatten the inputs and un-pool
#         x = tf.scatter_nd(index, K.flatten(x), K.cast(flat_output_shape, 'int64'))
#         # reshape the output in the correct shape
#         x = K.reshape(x, outs)

#         # update the integer shape of the Keras Tensor
#         ins = K.int_shape(idx)
#         x.set_shape([ins[0], ins[1] * pool_size[0], ins[2] * pool_size[1], ins[3]])

#         return x

class MaxPooling2DVarPropagationLayer(VarPropagationLayer):

    def __init__(self, pooling_layer, use_cov=False, **kwargs):
        self.idx = None
        super(MaxPooling2DVarPropagationLayer, self).__init__(pooling_layer, use_cov=False, **kwargs)

    def _call_diag_cov(self, x):
        pooled, self.idx = self._pool2d_argmax(self.layer.input, 
                                          pool_size=self.layer.pool_size,
                                          strides=self.layer.strides,
                                          padding=self.layer.padding)
        shape_x = tf.shape(x)
        shape_pooled = tf.shape(pooled)
        shape_arg_max = tf.shape(self.idx)
        x_flat = tf.reshape(x, [shape_x[0], shape_x[1]*shape_x[2]*shape_x[3]])
        arg_max_flat = tf.reshape(self.idx, [shape_arg_max[0], shape_arg_max[1]*shape_arg_max[2]*shape_arg_max[3]])
        arg_max_flat = tf.cast(arg_max_flat, tf.int32)
        out = tf.batch_gather(x_flat, arg_max_flat)
        out = tf.reshape(out, [shape_pooled[0], shape_pooled[1], shape_pooled[2], shape_pooled[3]])
        return out
    
    def _pool2d_argmax(self, x: 'Tensor', pool_size: tuple,
        strides: tuple=(1, 1),
        padding: str='valid',
        data_format: str=None
        ) -> tuple:
        """
        2D Pooling that returns indexes too.
        Args:
            x: Tensor or variable.
            pool_size: tuple of 2 integers.
            strides: tuple of 2 integers.
            padding: string, `"same"` or `"valid"`.
            data_format: string, `"channels_last"` or `"channels_first"`.
        Returns:
            A tensor, result of 2D pooling.
        Raises:
            ValueError: if `data_format` is neither `"channels_last"` or `"channels_first"`.
        """
        # get the normalized data format
        data_format = K.common.normalize_data_format(data_format)
        # pre-process the input tensor
        x, tf_data_format = _preprocess_conv2d_input(x, data_format)
        padding = _preprocess_padding(padding)
        # update strides and pool size based on data format
        if tf_data_format == 'NHWC':
            strides = (1,) + strides + (1,)
            pool_size = (1,) + pool_size + (1,)
        else:
            strides = (1, 1) + strides
            pool_size = (1, 1) + pool_size
        # get the values and the indexes from the max pool operation
        x, idx = tf.nn.max_pool_with_argmax(x, pool_size, strides, padding)
        # update shapes if necessary
        if data_format == 'channels_first' and tf_data_format == 'NHWC':
            # NHWC -> NCHW
            x = tf.transpose(x, (0, 3, 1, 2))
            # NHWC -> NCHW
            idx = tf.transpose(idx, (0, 3, 1, 2))

        return x, idx
