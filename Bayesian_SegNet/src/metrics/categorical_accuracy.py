"""A metric to calculate categorical accuracy."""
import numpy as np
from keras import backend as K
from ..backend.tensorflow_backend import confusion_matrix


def build_categorical_accuracy(weights=None):
    """
    Build a categorical accuracy method using given weights.

    Args:
        weights: the weights to use for the metric

    Returns:
        a callable categorical accuracy evaluation metric

    """
    def categorical_accuracy(y_true, y_pred):
        """
        Return a categorical accuracy tensor for label and prediction tensors.

        Args:
            y_true: the ground truth labels to compare against
            y_pred: the predicted labels from a loss network

        Returns:
            a tensor of the categorical accuracy between truth and predictions

        """
        # get number of labels to calculate IoU for
        num_classes = K.int_shape(y_pred)[-1]
        # set the weights to all 1 if there are none specified
        _weights = np.ones(num_classes) if weights is None else weights
        # convert the one-hot tensors into discrete label tensors with ArgMax
        y_true = K.flatten(K.argmax(y_true, axis=-1))
        y_pred = K.flatten(K.argmax(y_pred, axis=-1))
        # calculate the confusion matrix of the ground truth and predictions
        confusion = confusion_matrix(y_true, y_pred, num_classes=num_classes)
        # confusion will return integers, but we need floats to multiply by eye
        confusion = K.cast(confusion, K.floatx())
        # extract the number of correct guesses from the diagonal
        correct = _weights * K.sum(confusion * K.eye(num_classes), axis=-1)
        # extract the number of total values per class from ground truth
        total = _weights * K.sum(confusion, axis=-1)
        # calculate the total accuracy
        return K.sum(correct) / K.sum(total)

    return categorical_accuracy


# explicitly define the outward facing API of this module
__all__ = [build_categorical_accuracy.__name__]
