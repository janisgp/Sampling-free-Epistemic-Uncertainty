"""Methods to get unwrapped predictions from different architectures."""
import numpy as np
from .utils import extract_aleatoric
from .utils import heatmap


def predict_epistemic(uncertainty_propagator, generator, camvid) -> tuple:
    """
    Return post-processed predictions for the given generator.

    Args:
        uncertainty_propagator: the uncertainty_propagator to use to predict with
        generator: the generator to get data from
        camvid: the CamVid instance for un-mapping target values

    Returns:
        a tuple of for NumPy tensors with RGB data:
        - the batch of RGB X values
        - the unmapped RGB batch of y values
        - the unmapped RGB predicted mean values from the model
        - the heat-map RGB values of the epistemic uncertainty

    """
    if isinstance(generator, np.ndarray):
        # predict mean values and variance
        y_pred, sigma = uncertainty_propagator.predict(generator)
        # return X values, unmapped y and u values, and heat-map of sigma**2
        return generator, camvid.unmap(y_pred), heatmap(sigma)

    # get the batch of data
    imgs, y_true = next(generator)
    # predict mean values and variance
    y_pred, sigma = uncertainty_propagator.predict(imgs)
    # return X values, unmapped y and u values, and heat-map of sigma**2
    return imgs, camvid.unmap(y_true), camvid.unmap(y_pred), heatmap(sigma)

def predict_epistemic_all(uncertainty_propagator, generator, camvid) -> tuple:
    """
    Return post-processed predictions for the given generator.

    Args:
        uncertainty_propagator: the uncertainty_propagator to use to predict with
        generator: the generator to get data from
        camvid: the CamVid instance for un-mapping target values

    Returns:
        a tuple of for NumPy tensors with RGB data:
        - the batch of RGB X values
        - the unmapped RGB batch of y values
        - the unmapped RGB predicted mean values from the model
        - the heat-map RGB values of the epistemic uncertainty

    """
    if isinstance(generator, np.ndarray):
        # predict mean values and variance
        mc_mode = uncertainty_propagator._mc_mode
        res = dict()
        for m in [False, True]:
            uncertainty_propagator.set_mc_mode(m)
            y_pred, sigma = uncertainty_propagator.predict(generator)
            if m:
                res['mc'] = [camvid.unmap(y_pred), heatmap(sigma)]
            else:
                res['approx'] = [camvid.unmap(y_pred), heatmap(sigma)]
        uncertainty_propagator.set_mc_mode(mc_mode)
        # return X values, unmapped y and u values, and heat-map of sigma**2
        return generator, res

    # get the batch of data
    imgs, y_true = next(generator)
    # predict mean values and variance
    y_pred, sigma = uncertainty_propagator.predict(imgs)
    mc_mode = uncertainty_propagator._mc_mode
    res = dict()
    for m in [False, True]:
        uncertainty_propagator.set_mc_mode(m)
        y_pred, sigma = uncertainty_propagator.predict(imgs)
        if m:
            res['mc'] = [camvid.unmap(y_pred), heatmap(sigma)]
        else:
            res['approx'] = [camvid.unmap(y_pred), heatmap(sigma)]
    uncertainty_propagator.set_mc_mode(mc_mode)
    # return X values, unmapped y and u values, and heat-map of sigma
    return imgs, camvid.unmap(y_true), res