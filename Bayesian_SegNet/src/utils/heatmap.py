"""A method to map vectors to color maps."""
import numpy as np
from matplotlib import pyplot as plt


def heatmap(arr: np.ndarray,
    color_map: str='binary',
    normalize: bool=True,
) -> np.ndarray:
    """
    Use the given color map to convert the input vector to a heat-map.

    Args:
        arr: the vector to convert to an RGB heat-map
        color_map: the color map to use
        normalize: whether to normalize the values before using them

    Returns:
        arr mapped to RGB using the given color map (vector of bytes)

    """
    # normalize the values if the flag is on
    if normalize:
        arr = plt.Normalize()(arr)
    # unwrap the color map from matplotlib
    color_map = plt.cm.get_cmap(color_map)
    # get the heat-map from the color map in RGB (i.e., omit the alpha channel)
    _heatmap = color_map(arr)[..., :-1]
    # scale heat-map from [0,1] to [0, 255] as a vector of bytes
    _heatmap = (255 * _heatmap).astype(np.uint8)

    return _heatmap


# explicitly define the outward facing API of this module
__all__ = [heatmap.__name__]
