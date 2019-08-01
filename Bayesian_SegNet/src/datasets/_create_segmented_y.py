"""A method to create a segmented version of an RGB dataset."""
import os
import glob
import shutil
import hashlib
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def _hash(mapping: dict) -> str:
    """
    Return a hash of an entire dictionary.

    Args:
        mapping: a dictionary of hash-able keys to hash-able values, i.e.,
                 __str__ should return a unique representation of each object

    Returns:
        a hash of the dictionary

    """
    # create a string to store the mapping in
    mapping_str = ''
    # iterate over the sorted dictionary keys to ensure reproducibility
    for key in sorted(mapping.keys()):
        # add the key value pairing to the string representation
        mapping_str += '{}{}'.format(key, mapping[key])
    # convert the string to bytes and return the MD5 has of the bytes
    return hashlib.md5(bytes(mapping_str, 'utf8')).hexdigest()


def _setup_directories(path: str, mapping: dict, overwrite: bool) -> tuple:
    """
    Setup directories for the pre-processed y data.

    Args:
        path: the path to the directory to setup the output data structure in
        mapping: the mapping dictionary to use to hash an output directory
        overwrite: whether to overwrite if the data already exists

    Returns:
        a string pointing to the output path or a tuple of new directories:
        -   the glob path for loading y data
        -   the output directory
        -   the path to the pre-processed training data
        -   the name of the new y directory
        -   the path to the metadata file for the pre-processed data

    """
    y_glob = os.path.join(path, 'y/**/**/*.png')
    # create the output directory for the y data
    if mapping is None:
        new_y_dir = 'y_full'
    else:
        # use the mapping dictionary as a hash to locate its files on disk
        new_y_dir = 'y_{}'.format(_hash(mapping))
    # join the current directory to the output y directory to form output path
    output_dir = os.path.join(path, new_y_dir)
    # check if the metadata file exists (data is corrupt if missing)
    metadata_filename = os.path.join(output_dir, 'metadata.csv')
    # check if the metadata file exists to return early (already processed)
    if os.path.isfile(metadata_filename) and not overwrite:
        return output_dir
    # create a list of directories with each subset of the data
    y_subsets = os.path.join(path, 'y')
    subsets = os.listdir(y_subsets)
    data_dirs = {subset: os.path.join(output_dir, subset) for subset in subsets}
    # ensure that a train subset exists
    if 'train' not in data_dirs.keys():
        raise ValueError('no `train` subset found in {}'.format(y_subsets))
    # create all necessary directories (overwrite existing directories)
    for data_dir in data_dirs.values():
        shutil.rmtree(data_dir, ignore_errors=True)
        os.makedirs(os.path.join(data_dir, 'data'))

    return y_glob, output_dir, data_dirs['train'], new_y_dir, metadata_filename


def _rgb_to_onehot(metadata: pd.DataFrame, arb_img_file: str) -> 'Callable':
    """
    Create a method to map RGB images to one-hot vectors.

    Args:
        metadata: the metadata table to get the mapping of RGB to codes
        arb_img_file: the path to an arbitrary image to understand shape

    Returns:
        a callable method to map images to one-hot NumPy tensors

    """
    # create a vectorized method to convert RGB points to discrete codes
    codes = metadata[['rgb', 'code']].set_index('rgb')['code'].to_dict()
    # convert the RGB tuples to 24-bit (int32) keys
    codes = {(k[0] << 16) + (k[1] << 8) + k[2]: v for (k, v) in codes.items()}
    rgb_to_code = np.vectorize(codes.get, otypes=['object'])
    # get the code for the Void label to use for invalid pixels
    void_code = metadata[metadata['label'] == 'Void'].code
    # determine the number of labels and create the identity matrix
    identity = np.eye(len(metadata['label_used'].unique()))

    def rgb_to_onehot(img_file: str, output_dtype: np.dtype) -> np.ndarray:
        """
        Convert an RGB image to a NumPy one-hot with given output type.

        Args:
            img_file: the path to the RGB image on disk
            output_dtype: the dtype of the one-hot tensor to create

        Returns:
            a one-hot representation of the image stored at img_file

        """
        # load the data as a NumPy array
        with Image.open(img_file) as raw_img:
            img = np.array(raw_img.convert('RGB'))
        # create a map to shift images left (to convert to hex)
        red = np.full(img.shape[:-1], 16)
        green = np.full(img.shape[:-1], 8)
        blue = np.zeros(img.shape[:-1])
        # stack the shift matrices into a tensor to shift tensors of RGB data
        left = np.stack([red, green, blue], axis=-1).astype(int)
        # convert the image to hex and decode its discrete values
        discrete = rgb_to_code(np.left_shift(img, left).sum(axis=-1))
        # check that each pixel has been overwritten
        invalid = discrete == None
        if invalid.any():
            template = 'WARNING: {} invalid pixels in {}'
            print(template.format(invalid.sum(), img_file))
            discrete[invalid] = void_code
        # convert the discrete mapping to a one hot encoding
        return identity[discrete.astype(int)].astype(output_dtype)

    return rgb_to_onehot


def _class_and_file_totals(subset: str) -> pd.DataFrame:
    """
    Calculate class statistics for a subset of data.

    Args:
        subset: the path to subset of data to count values in

    Returns:
        a DataFrame with columns for class totals and file totals in pixels

    """
    # create a glob to reference all the compressed NumPy files in the dataset
    files = glob.glob(os.path.join(subset, 'data/*.npz'))
    shape = np.load(files[0])['y'].shape
    # create placeholders for class and file totals
    pixels = pd.Series(np.zeros(shape[-1]), name='pixels')
    pixels_total = pd.Series(np.zeros(shape[-1]), name='pixels_total')
    # iterate over the files in the dataset
    for file in files:
        # load the target one-hot from the file
        onehot = np.load(file)['y']
        # get the class totals for the file (i.e., number of pixels per class)
        pixels += onehot.sum(axis=(0, 1))
        # get the file total for the file (i.e., true if a pixel present)
        pixels_total += onehot.any(axis=(0, 1))
    # multiply the file totals count by the number of pixels per file
    pixels_total *= np.prod(shape[:-1])
    # create a DataFrame from the two columns of data
    return pd.DataFrame([pixels, pixels_total]).T.astype(int)


def create_segmented_y(
    path: str,
    metadata: pd.DataFrame,
    mapping: dict=None,
    output_dtype: str='uint8',
    overwrite: bool=False,
) -> str:
    """
    Create a segmented version of an RGB dataset.

    Args:
        path: the path housing the y data to create the pre-processed data in
        metadata: the metadata describing the mapping between RGB and codes
        mapping: a dictionary mapping existing values to new ones for
                 dimensionality reduction
        output_dtype: the dtype of the output NumPy array of values
        overwrite: whether to overwrite the data if it already exists

    Returns:
        the path to the output directory created by the method

    """
    # setup the directories for the pre-processed data
    dirs = _setup_directories(path, mapping, overwrite)
    # if dirs is a string, then overwrite is off and the dataset exists. return
    # the path to the output directory with the pre-processed data
    if isinstance(dirs, str):
        return dirs

    # unpack the tuple of directories
    y_glob, output_dir, train, new_y_dir, metadata_filename = dirs
    # create a vectorized method to convert RGB to one-hot tensors
    rgb_to_onehot = _rgb_to_onehot(metadata, glob.glob(y_glob)[0])

    # iterate over all the files in the source directory
    for img_file in tqdm(sorted(glob.glob(y_glob)), unit='image'):
        # replace the y directory with the new directory name
        output_file = img_file.replace('/y/', '/' + new_y_dir + '/')
        # replace the file type for NumPy
        output_file = output_file.replace('.png', '.npz')
        # convert the image to hex and decode its discrete values
        onehot = rgb_to_onehot(img_file, output_dtype)
        # save the file to its output location
        np.savez_compressed(output_file, y=onehot)

    # calculate the number of pixels belonging to each class in training
    weights = _class_and_file_totals(train)
    weights.to_csv(os.path.join(output_dir, 'weights.csv'))
    # save the metadata to disk for working with the encoded data. this file
    # is used as a check to ensure that the above process completed
    metadata.to_csv(metadata_filename, index=False)

    return output_dir


# explicitly define the outward facing API of this module
__all__ = [create_segmented_y.__name__]
