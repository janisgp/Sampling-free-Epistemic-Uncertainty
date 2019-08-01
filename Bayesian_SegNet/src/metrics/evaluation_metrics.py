"""Methods to evaluate model outputs using NumPy."""
import numpy as np
from sklearn.metrics import confusion_matrix


def accuracy(confusion: np.ndarray, mask: np.ndarray) -> float:
    """
    Return the global accuracy from a confusion matrix.

    Args:
        confusion: the confusion matrix between ground truth and predictions
        mask: a mask for ignoring specific labels

    Returns:
        a float representing the global accuracy in the confusion matrix

    """
    # extract the number of correct guesses from the diagonal
    preds_correct = mask * np.sum(confusion * np.eye(len(confusion)), axis=-1)
    # extract the number of total values per class from ground truth
    trues = mask * np.sum(confusion, axis=-1)
    # calculate the total accuracy
    return np.sum(preds_correct) / np.sum(trues)


def class_accuracy(confusion: np.ndarray) -> np.ndarray:
    """
    Return the per class accuracy from confusion matrix.

    Args:
        confusion: the confusion matrix between ground truth and predictions

    Returns:
        a vector representing the per class accuracy

    """
    # extract the number of correct guesses from the diagonal
    preds_correct = np.sum(confusion * np.eye(len(confusion)), axis=-1)
    # extract the number of total values per class from ground truth
    trues = np.sum(confusion, axis=-1)
    # get per class accuracy by dividing correct by total
    return preds_correct / trues


def iou(confusion: np.ndarray) -> np.ndarray:
    """
    Return the per class Intersection over Union (I/U) from confusion matrix.

    Args:
        confusion: the confusion matrix between ground truth and predictions

    Returns:
        a vector representing the per class I/U

    Reference:
        https://en.wikipedia.org/wiki/Jaccard_index

    """
    # get |intersection| (AND) from the diagonal of the confusion matrix
    intersection = (confusion * np.eye(len(confusion))).sum(axis=-1)
    # calculate the total ground truths and predictions per class
    preds = confusion.sum(axis=0)
    trues = confusion.sum(axis=-1)
    # get |union| (OR) from the predictions, ground truths, and intersection
    union = trues + preds - intersection
    # return the intersection over the union
    return intersection / union


def metrics(confusion: np.ndarray, mask: np.ndarray) -> tuple:
    """
    Return metrics evaluating a categorical classification task.

    Args:
        confusion: the confusion matrix between ground truth and predictions
        mask: the mask to use for the metrics

    Returns:
        a tuple of metrics for evaluation:
        -   float: the global accuracy
        -   float: the mean per class accuracy
        -   float: the mean I/U
        -   ndarray: a vector representing the per class I/U

    """
    # set the mask to all 1 if there are none specified
    mask = np.ones(y_pred.shape[-1]) if mask is None else mask
    # calculate the global accuracy from the confusion matrix
    _accuracy = accuracy(confusion, mask)
    # calculate the class accuracies from the confusion matrix
    _class_accuracy = class_accuracy(confusion)[mask.astype(bool)]
    # take the mean of the class accuracies
    mean_per_class_accuracy = _class_accuracy.mean()
    # calculate the per class IoUs from the confusion matrix
    _iou = iou(confusion)[mask.astype(bool)]
    # take the mean of the per class I/Us
    mean_iou = _iou.mean()

    return _accuracy, mean_per_class_accuracy, mean_iou, _iou


# explicitly define the outward facing API of this module
__all__ = [metrics.__name__]
