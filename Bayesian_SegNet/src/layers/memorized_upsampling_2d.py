"""A 2D up-sampling layer that uses indexes from memorized pooling."""
from keras.layers import UpSampling2D
from ..backend.tensorflow_backend import unpool2d_argmax


class MemorizedUpsampling2D(UpSampling2D):
    """A 2D up-sampling layer that uses indexes from memorized pooling."""

    def __init__(self, *args, pool, **kwargs):
        """
        Initialize a new up-sampling layer using memorized down-sample index.

        Args:
            pool: the memorized index form pool2d_argmax

        Returns:
            None

        """
        # get the size from the pooling operation
        size = pool.pool_size
        # call the super constructor
        super(MemorizedUpsampling2D, self).__init__(*args, size=size, **kwargs)
        # set the index from the pool index calculation
        self.idx = pool.idx

    def call(self, inputs):
        # up-sample the inputs using the indexes from the max operation in
        # pooling of a certain size
        return unpool2d_argmax(inputs, self.idx, self.size)


# explicitly define the outward facing API of this module
__all__ = [MemorizedUpsampling2D.__name__]
