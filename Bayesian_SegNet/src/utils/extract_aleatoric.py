"""A method to extract the Aleatoric uncertainty for predicted class labels."""
import numpy as np


def extract_aleatoric(sigma: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Extract the Aleatoric uncertainty from sigma and predicted y outputs.

    Args:
        sigma: the standard deviation predicted by the loss network
        y_pred: the predicted values as either a one-hot or Softmax output

    Returns:
        a 2D batch tensor of aleatoric uncertainties

    """
    # use ogrid to index sigma using the integer class labels
    ogrid = np.ogrid[0:y_pred.shape[0], 0:y_pred.shape[1], 0:y_pred.shape[2]]
    batch, height, width = ogrid
    # use fancy indexing to extract the value at the index of each class label
    sigma2 = np.square(sigma)
    aleatoric = sigma2[batch, height, width, np.argmax(y_pred, axis=-1)]

    return aleatoric


# explicitly define the outward facing API of this module
__all__ = [extract_aleatoric.__name__]
