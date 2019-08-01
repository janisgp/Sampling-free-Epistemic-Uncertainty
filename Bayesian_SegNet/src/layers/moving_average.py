"""A layer to calculate an Exponential Moving Average (EMA)."""
from keras import backend as K
from keras.layers import Layer


class MovingAverage(Layer):
    """A layer to calculate an Exponential Moving Average (EMA)."""

    def __init__(self, momentum=0.9, **kwargs):
        """
        Initialize a new repeat tensor layer.

        Args:
            momentum: the momentum of the moving average
            kwargs: keyword arguments for the super constructor

        Returns:
            None

        """
        # initialize with the super constructor
        super(MovingAverage, self).__init__(**kwargs)
        # store the instance variables of this layer
        self.momentum = momentum

    def call(self, inputs):
        """
        Forward pass through the layer.

        Args:
            inputs: the tensor to perform the stack operation on
            training: whether the layer is in the training phase

        Returns:
            the input tensor stacked self.n times along axis 1

        """
        # initialize the average with zeros
        average = K.zeros((1, ) + K.int_shape(inputs)[1:])
        # update the average using an exponential update
        average = self.momentum * inputs + (1 - self.momentum) * average

        return average


# explicitly define the outward facing API of this module
__all__ = [MovingAverage.__name__]
