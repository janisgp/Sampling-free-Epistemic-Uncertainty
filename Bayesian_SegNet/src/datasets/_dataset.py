"""A class for interacting with datasets in this project."""
import ast
import os
import numpy as np
import pandas as pd
from ._create_segmented_y import create_segmented_y
from ._generators import CropImageDataGenerator
from ._generators import CropNumpyDataGenerator
from ._generators import repeat_generator
from ._label_colors import load_label_metadata


class DataSet(object):
    """An instance of a DataSet."""

    # create a constant path for the dataset to reference
    PATH = None

    # the default mapping file
    DEFAULT_MAPPING = None

    def __init__(self,
        mapping: dict=None,
        ignored_labels: list=['Void'],
        x_repeats: int=0,
        y_repeats: int=0,
        target_size: tuple=(720, 960),
        crop_size: tuple=(224, 224),
        horizontal_flip: bool=False,
        vertical_flip: bool=False,
        batch_size: int=3,
        shuffle: bool=True,
        seed: int=1,
    ) -> None:
        """
        Initialize a new CamVid dataset instance.

        Args:
            mapping: mapping to use when generating the preprocessed targets
            ignored_labels: a list of string label names to ignore (0 weight)
            x_repeats: the number of times to repeat the output of x generator
            y_repeats: the number of times to repeat the output of y generator
            target_size: the image size of the dataset
            crop_size: the size to crop images to. if None, apply no crop
            horizontal_flip: whether to randomly flip images horizontally
            vertical_flip whether to randomly flip images vertically
            batch_size: the number of images to load per batch
            shuffle: whether to shuffle images in the dataset
            seed: the random seed to use for the generator

        Returns:
            None

        """

        # locate the X and y directories
        self._x = os.path.join(self.PATH, 'X')
        metadata = load_label_metadata(self.PATH, mapping)
        self._y = create_segmented_y(self.PATH, metadata, mapping)
        # store remaining keyword arguments
        self.ignored_labels = ignored_labels
        self.x_repeats = x_repeats
        self.y_repeats = y_repeats
        self.target_size = target_size
        self.crop_size = crop_size
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        # create a vectorized method to map discrete codes to RGB pixels
        self._unmap = np.vectorize(self.discrete_to_rgb_map.get)

    @property
    def n(self) -> int:
        """Return the number of training classes in this dataset."""
        return len(self.metadata['code'].unique())

    @property
    def class_weights(self) -> dict:
        """Return a dictionary of class weights keyed by discrete label."""
        weights = pd.read_csv(os.path.join(self._y, 'weights.csv'), index_col=0)
        # calculate the frequency of each class (and swap to NumPy)
        freq = (weights['pixels'] / weights['pixels_total']).values
        # ignore the ignored values when calculating median
        med = np.median(np.delete(freq, self.ignored_codes))
        # calculate the weights as the median frequency divided by all freq
        weights = (med / freq)
        # set ignored weights to 0
        weights[self.ignored_codes] = 0

        return weights

    @property
    def class_mask(self) -> dict:
        """Return a dictionary of class weights keyed by discrete label."""
        weights = self.class_weights
        # get the class mask as a boolean vector
        class_mask = weights > 0
        # cast the boolean vector to integers for math
        class_mask = class_mask.astype(weights.dtype)

        return class_mask

    def data_gen_args(self, context: str) -> dict:
        """
        Return the keyword arguments for creating a new data generator.

        Args:
            context: the context for the call (i.e., train for training)

        Returns:
            a dictionary of keyword arguments to pass to DataGenerator.__init__

        """
        # return for training
        if context == 'train':
            return dict(
                horizontal_flip=self.horizontal_flip,
                vertical_flip=self.vertical_flip,
                image_size=self.crop_size
            )
        # return for validation / testing (i.e., inference)
        return dict(image_size=self.crop_size)

    def flow_args(self, context: str) -> dict:
        """
        Return the keyword arguments for flowing from a data generator.

        Args:
            context: the context for the call (i.e., train for training)

        Returns:
            a dictionary of keyword arguments to pass to flow_from_directory

        """
        # return for training
        if context == 'train':
            return dict(
                batch_size=self.batch_size,
                class_mode=None,
                target_size=self.target_size,
                shuffle=self.shuffle,
                seed=self.seed
            )
        # return for validation / testing (i.e., inference)
        return dict(
            batch_size=1,
            class_mode=None,
            target_size=self.target_size,
            seed=self.seed
        )

    @property
    def metadata(self) -> pd.DataFrame:
        """Return the metadata associated with this dataset."""
        return pd.read_csv(os.path.join(self._y, 'metadata.csv'))

    def _discrete_dict(self, col: str) -> dict:
        """
        Return a dictionary mapping discrete codes to values in another column.

        Args:
            col: the name of the column to map discrete code values to

        Returns:
            a dictionary mapping unique codes to values in the given column

        """
        return self.metadata[['code', col]].set_index('code').to_dict()[col]

    @property
    def discrete_to_rgb_map(self) -> dict:
        """Return a dictionary mapping discrete codes to RGB pixels."""
        rgb_draw = self._discrete_dict('rgb_draw')
        # convert the strings in the RGB draw column to tuples
        return {k: ast.literal_eval(v) for (k, v) in rgb_draw.items()}

    @property
    def discrete_to_label_map(self) -> dict:
        """Return a dictionary mapping discrete codes to RGB pixels."""
        return self._discrete_dict('label_used')

    @property
    def label_to_discrete_map(self) -> dict:
        """Return a dictionary mapping discrete codes to RGB pixels."""
        return {v: k for (k, v) in self.discrete_to_label_map.items()}

    @property
    def ignored_codes(self) -> list:
        """Return a list of the ignored discrete coded labels."""
        # turn the label to discrete code map into a vectorized function
        get = np.vectorize(self.label_to_discrete_map.get, otypes=['uint64'])
        # unwrap the codes for each label in the ignored labels list
        ignored = get(self.ignored_labels)
        # return the ignored codes
        return list(ignored)

    def unmap(self, y_discrete: np.ndarray) -> np.ndarray:
        """
        Un-map a one-hot vector y frame to the target RGB values.

        Args:
            y_discrete: the one-hot vector to convert to an RGB image

        Returns:
            an RGB encoding of the one-hot input tensor

        """
        return np.stack(self._unmap(y_discrete.argmax(axis=-1)), axis=-1)

    def generators(self) -> dict:
        """Return a dictionary with both training and validation generators."""
        # the dictionary to hold generators by key value (training, validation)
        generators = dict()
        # iterate over the generator subsets
        for subset in next(os.walk(self._y))[1]:
            # make sure file is a directory (in case it's .DS_Store or similar)
            if subset in {'.DS_Store'}:
                continue
            # create generators to load images (X) and NumPy tensors (y)
            x_g = CropImageDataGenerator(**self.data_gen_args(subset))
            y_g = CropNumpyDataGenerator(**self.data_gen_args(subset))
            # get the path for the subset of data
            _x = os.path.join(self._x, subset)
            _y = os.path.join(self._y, subset)
            # combine X and y generators into a single generator with repeats
            generators[subset] = repeat_generator(
                x_g.flow_from_directory(_x, **self.flow_args(subset)),
                y_g.flow_from_directory(_y, **self.flow_args(subset)),
                x_repeats=self.x_repeats,
                y_repeats=self.y_repeats,
            )

        return generators

    @classmethod
    def load_mapping(cls, mapping_file: str=None, sep: str=r'\s+') -> dict:
        """
        Load a mapping file from disk as a dictionary.

        Args:
            mapping_file: file pointing to a text file with mapping data
            sep: the separator for entries in the file

        Returns:
            a dictionary mapping old classes to generalized classes

        """
        if mapping_file is None:
            if cls.DEFAULT_MAPPING is None:
                raise ValueError('no default mapping to use!')
            mapping_file = cls.DEFAULT_MAPPING
        # the names of the columns in the file
        names = ['og', 'new']
        # load the DataFrame with the original classes as the index col
        mapping = pd.read_table(mapping_file,
            sep=sep,
            names=names,
            index_col='og'
        )
        # return a dict of the new column mapping old classes to new classes
        return mapping['new'].to_dict()


# explicitly define the outward facing API of this module
__all__ = [DataSet.__name__]
