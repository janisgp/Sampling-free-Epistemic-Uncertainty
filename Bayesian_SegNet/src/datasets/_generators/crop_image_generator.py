"""An Image Generator extension to crop images to a given size."""
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from .numpy_data_generator import NumpyDataGenerator


def crop_dim(dim: int, crop_size: int) -> tuple:
    """
    Return the crop bounds of a dimension using RNG.

    Args:
        dim: the value of the dimension
        crop_size: the value to crop the dimension to

    Returns:
        a tuple of:
        -   the starting point of the crop
        -   the stopping point of the crop

    """
    # if crop size is equal to input size, return the input dimension
    if crop_size == dim:
        return 0, dim
    # otherwise generate a random anchor point and add the crop size to it
    dim_0 = np.random.randint(0, dim - crop_size)
    dim_1 = dim_0 + crop_size
    return dim_0, dim_1


def random_crop(tensor: np.ndarray, image_size: tuple) -> np.ndarray:
    """
    Return a random crop of a tensor.

    Args:
        tensor: the tensor to get a random crop from
        image_size: the size of the cropped box to return

    Returns:
        a random crop of the tensor with shape image_size

    """
    # crop the height
    h_0, h_1 = crop_dim(tensor.shape[0], image_size[0])
    # crop the width
    w_0, w_1 = crop_dim(tensor.shape[1], image_size[1])
    # return the cropped tensor
    return tensor[h_0:h_1, w_0:w_1]


class CropDataGenerator(object):
    """An Image Generator extension to crop tensors to a given size."""

    def __init__(self, *args, image_size=None, **kwargs) -> None:
        """
        Create a new Segment Image Data generator.

        Args:
            *args: positional arguments for the ImageDataGenerator super class
            image_size: the image size to crop to
            **kwargs: keyword arguments for the ImageDataGenerator super class

        Returns:
            None

        """
        if image_size is not None and not isinstance(image_size, tuple):
            raise TypeError('image_size should be of type: tuple')
        super().__init__(*args, **kwargs)
        self.image_size = image_size

    def apply_transform(self, *args, **kwargs):
        """Apply a transform to the input tensor with given parameters."""
        # get the batch from the super transformer first
        batch = super().apply_transform(*args, **kwargs)
        # map this batch of items to output dimension
        if self.image_size is not None:
            return random_crop(batch, self.image_size)
        return batch

    def flow_from_directory(self, *args, **kwargs):
        """Create a directory iterator to load from."""
        # get the directory iterator from the super call
        iterator = super().flow_from_directory(*args, **kwargs)
        # change the output dimension of the iterator to support the new
        # number of channels defined by the transformers length
        if self.image_size is not None:
            iterator.image_shape = (*self.image_size, iterator.image_shape[-1])
        return iterator


class CropImageDataGenerator(CropDataGenerator, ImageDataGenerator):
    """An Image Generator extension to crop images to a given size."""


class CropNumpyDataGenerator(CropDataGenerator, NumpyDataGenerator):
    """An Image Generator extension to crop NumPy tensors to a given size."""


# explicitly define the outward facing API of this module
__all__ = [
    CropImageDataGenerator.__name__,
    CropNumpyDataGenerator.__name__,
]
