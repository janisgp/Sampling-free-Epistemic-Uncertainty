"""A layer to calculate the entropy of a tensor."""
from keras.layers import Layer
from keras import backend as K


class Entropy(Layer):
    """A layer to calculate the entropy of a tensor."""

    def compute_output_shape(self, input_shape):
        """
        Return the output shape of the layer for given input shape.

        Args:
            input_shape: the input shape to transform to output shape

        Returns:
            the output shape as a function of input shape (reduced by 1 dim)

        """
        return input_shape[:-1]

    def call(self, inputs, **kwargs):
        """
        Forward pass through the layer.

        Args:
            inputs: the tensor to calculate the entropy over
            **kwargs: extra keyword arguments

        Returns:
            the tensor reduced by one dimension representing entropy

        """
        return -1 * K.sum(K.log(inputs) * inputs, axis=-1)


# explicitly define the outward facing API of this module
__all__ = [Entropy.__name__]
