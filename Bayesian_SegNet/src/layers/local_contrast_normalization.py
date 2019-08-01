"""A Keras layer to normalize images using local contrast normalization."""
import keras.backend as K
from keras.layers.core import Layer


def conv2d(inputs, kernel):
    """
    Convolve over the inputs using the given kernel.

    Args:
        inputs: the inputs to convolve
        kernel: the kernel to use in the convolution

    Returns:
        the output from the convolution operation over inputs using the kernel

    """
    # convolve over the inputs using the kernel with same shape padding
    channels = []
    # iterate over the channels in the image
    for i in range(K.int_shape(inputs)[-1]):
        # append the convolved channel to the output tensor
        channels += [K.conv2d(inputs[..., i:i+1], kernel, padding='same')]
    # take the mean values over the concatenated tensor of channels
    channels = K.mean(K.concatenate(channels, axis=-1), axis=-1, keepdims=True)

    return channels


def normal_kernel(kernel_size, mean=1.0, scale=0.05):
    """
    Return a new Gaussian RGB kernel with given layer size.

    Args:
        kernel_size: the size of the kernel
        mean: the mean for the Gaussian randomness
        scale: the scale for the Gaussian randomness

    Returns:
        a Gaussian RGB kernel normalized to sum to 1

    """
    # create the kernel shape with square kernel, 1 expected input channel,
    # and 1 filter in total (i.e., 1 output channel)
    kernel_shape = (kernel_size, kernel_size, 1, 1)
    # create a random normal variable with given mean and scale
    kernel = K.random_normal(kernel_shape, mean=mean, stddev=scale)
    # normalize the values to ensure the sum of the filter is 1
    kernel = kernel / K.sum(kernel)

    return kernel


class LocalContrastNormalization(Layer):
    """A Keras layer to normalize images using local contrast normalization."""

    def __init__(self, kernel_size=9, **kwargs):
        """
        Initialize a new contrast normalization layer.

        Args:
            kernel_size: the size of the kernel to use in Gaussian kernels

        Returns:
            None

        """
        # type check kernel size
        try:
            kernel_size = int(kernel_size)
        except ValueError:
            raise TypeError('kernel_size must be an int')
        # ensure the kernel size is legal
        if kernel_size < 1:
            raise ValueError('kernel_size must be >= 1')
        # store kernel size
        self.kernel_size = kernel_size
        # call the super constructor
        super(LocalContrastNormalization, self).__init__(**kwargs)

    def call(self, inputs):
        """Forward pass through the contrast normalization layer."""
        # create a normal kernel (weighting window)
        kernel = normal_kernel(self.kernel_size)
        # perform subtractive normalization
        v = inputs - conv2d(inputs, kernel)
        # calculate sigma to perform divisive normalization
        sigma = K.sqrt(conv2d(K.square(v), kernel))
        mean = K.mean(sigma, axis=[1, 2])
        mean = K.expand_dims(K.expand_dims(mean, axis=1), axis=1)
        # perform the divisive normalization
        return v / K.maximum(mean, sigma)


# explicitly define the outward facing API of this module
__all__ = [LocalContrastNormalization.__name__]
