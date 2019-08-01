"""A pooling layer that memorized the indexes."""
from keras.layers import MaxPooling2D
from ..backend.tensorflow_backend import pool2d_argmax


class MemorizedMaxPooling2D(MaxPooling2D):
    """A max pooling layer that memorizes the indexes."""

    def __init__(self, *args, **kwargs):
        """Initialize a new Memorized Max Pooling 2D layer."""
        super(MemorizedMaxPooling2D, self).__init__(*args, **kwargs)
        self.idx = None

    def _pooling_function(self, inputs, pool_size, strides, padding, data_format):
        # get the output and indexes from the pool 2D with ArgMax method
        output, self.idx = pool2d_argmax(inputs, pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
        )
        # return the max pooling output
        return output


# explicitly define the outward facing API of this module
__all__ = [MemorizedMaxPooling2D.__name__]
