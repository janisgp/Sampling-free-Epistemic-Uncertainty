"""Methods to visualize data from the dataset."""
import numpy as np
from matplotlib import pyplot as plt


# the DPI for CamVid images is 72.0
DPI = 72.0


def plot_list(imgs: list, titles: list, figsize: tuple=(8,8)):
    
    for i in range(len(imgs)):
        plt.figure(figsize=(10,10))
        plt.title(titles[i], fontsize=24)
        plt.imshow(imgs[i]/255)
        plt.show()

def _plot(figure: plt.Figure, img: np.ndarray, title: str) -> None:
    """
    Plot the image on the given axis with a title.

    Args:
        figure: the figure to plot the image on
        img: the image to plot on the axes
        title: the title of the plot

    Returns:
        None

    """
    # plot the image
    figure.imshow(img.astype(np.uint8) / 255)
    # set the title for this subplot
    figure.set_title(title, fontsize=12)
    # remove the ticks from the x and y axes
    figure.xaxis.set_major_locator(plt.NullLocator())
    figure.yaxis.set_major_locator(plt.NullLocator())


def plot(dpi: float=256.0, order: list=None, **kwargs: dict) -> None:
    """
    Plot the original image, the true y, and an optional predicted y.

    Args:
        dpi: the DPI of the figure to render
        order: the order to plot the values in as a list of kwarg names
        kwargs: images to plot

    Returns:
        the figure holding the axes of the plot

    """
    # determine the figsize for plotting based on image_shape and DPI
    image_shape = list(kwargs.values())[0].shape
    figsize = image_shape[0] / DPI, image_shape[1] / DPI
    # if there is no order, iterate over all keyword args
    if order is None:
        # create subplots for each image
        fig, axarr = plt.subplots(len(kwargs), 1, figsize=figsize, dpi=dpi)
        # iterate over the images in the dictionary
        for idx, (title, img) in enumerate(kwargs.items()):
            _plot(axarr[idx], img, title)
    # if there is an order, plot the images in order
    else:
        # create subplots for each image
        fig, axarr = plt.subplots(len(order), 1, figsize=figsize, dpi=dpi)
        # iterate over the images in order
        for idx, title in enumerate(order):
            _plot(axarr[idx], kwargs[title], title)

    return fig


# explicitly define the outward facing API of this module
__all__ = [plot.__name__]
