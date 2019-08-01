"""A class for interacting with the CamVid data."""
import os
from .._dataset import DataSet


# a handle to the absolute path of this directory
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class CamVid(DataSet):
    """An instance of a CamVid dataset."""

    # create a constant path for the dataset to reference
    PATH = THIS_DIR

    # the default mapping file
    DEFAULT_MAPPING = os.path.join(THIS_DIR, '11_class.txt')


# explicitly define the outward facing API of this module
__all__ = [CamVid.__name__]
