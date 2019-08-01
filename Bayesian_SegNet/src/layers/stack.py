"""A layer to stack tensors along a new axis."""
from keras.layers import Layer
from keras import backend as K


class Stack(Layer):
    """A layer to stack tensors along a new axis."""

    def __init__(self, axis=-1, **kwargs):
        """
        Initialize a new stack tensor layer.

        Args:
            axis: the axis to expand the dimensions along
            kwargs: keyword arguments for the super constructor

        Returns:
            None

        """
        # initialize with the super constructor
        super(Stack, self).__init__(**kwargs)
        # store the instance variables of this layer
        self.axis = axis

    def call(self, inputs, **kwargs):
        """
        Forward pass through the layer.

        Args:
            inputs: a list of tensors to stack into one tensor
            **kwargs: extra keyword arguments

        Returns:
            the tensor with its dimensions expanded along self.axis

        """
        if not isinstance(inputs, list):
            raise TypeError('inputs must be a list of tensors')
        # set the stack size for the output shape computation
        # expand the inputs along the new dimension
        inputs = [K.expand_dims(t, axis=self.axis) for t in inputs]
        # concatenate the inputs along the new axis
        return K.concatenate(inputs, axis=self.axis)


# explicitly define the outward facing API of this module
__all__ = [Stack.__name__]
