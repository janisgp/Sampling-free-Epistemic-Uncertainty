"""A Method to evaluate segmentation models using NumPy metrics."""
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import confusion_matrix
from .metrics.evaluation_metrics import metrics
from keras.utils import to_categorical
import time


def probability_vs_uncertainty(uncertainty_propagator, generator, steps, bins: int=20):
    
    step_size = 100 // bins
    bins = np.arange(0, 100, step_size) / 100
    counts = np.zeros(bins.shape)
    sum_stds = np.zeros(bins.shape)
    
    for _ in tqdm(range(steps)):
        X, y = next(generator)
        preds, stds = uncertainty_propagator.predict(X)
        
        y_pred_argmax = preds.argmax(-1)
        
        idx = to_categorical(y_pred_argmax, num_classes=12).astype(np.bool)
        preds = np.reshape(preds[idx], preds.shape[:-1])
        
        for i in range(len(bins)):
            mask = (preds >= bins[i]) * (preds < bins[i] + step_size)
            counts[i] += stds[mask].size
            sum_stds[i] += stds[mask].sum()
            
    return bins, sum_stds/counts
    

def get_uncertainty_callibration(unc_pred_list: list):
    
    _unc = np.concatenate([el[0] for el in unc_pred_list], 0)
    _pred = np.concatenate([el[1] for el in unc_pred_list], 0)
    bins = np.arange(0, 1.1, 0.1)
    
    misclassification_rate = np.zeros(bins.shape)
    unc_pos_all = _unc[_pred]
    unc_neg_all = _unc[_pred == False]
    reps = 100
    samples = 100
    for i, b in enumerate(bins):
        for _ in range(reps):
            unc = 0
            nr_samples1, nr_samples2 = 0, 0
            if b < 1:
                nr_samples1 = int(samples*(1-b))
                unc += np.random.choice(unc_pos_all, size=nr_samples1).sum()
            if b > 0:
                nr_samples2 = int(samples*b)
                unc += np.random.choice(unc_neg_all, size=nr_samples2).sum()
            unc /= samples
            misclassification_rate[i] += unc
        misclassification_rate[i] /= reps
        
    return bins, misclassification_rate
    

def evaluate(uncertainty_propagator, generator, steps: int,
    mask: np.ndarray=None,
    code_map: dict=None,
    class_holdout: list=[]
) -> list:
    """
    Evaluate a segmentation model and return a DataFrame of metrics.

    Args:
        model: the model to generate predictions from
        generator: the generate to get data from
        steps: the number of steps to go through the generator
        mask: the mask to exclude certain labels
        code_map: a mapping of probability vector indexes to label names
        class_holdout: list holding indices of hold out classes

    Returns:
        a DataFrame with the metrics from the generator data

    """
    # get the number of classes from the output shape of the model
    out_s = uncertainty_propagator.model.output_shape
    num_classes = out_s[0][-1] if isinstance(out_s, list) else out_s[-1]
    # initialize a confusion matrix
    confusion = np.zeros((num_classes, num_classes))
    # iterate over the number of steps to generate data
    
    unc_pred_list = []
    unc_not_holdout, unc_holdout = [], [] 
    unc_of_class = np.zeros((num_classes))
    class_counts = np.zeros((num_classes))
    for step in tqdm(range(steps), unit='step'):
        # get the batch of data from the generator
        imgs, y_true = next(generator)
        nr_classes = y_true.shape[-1]
        
        # if y_true is a tuple or list, take the first model output
        y_true = y_true[0] if isinstance(y_true, (tuple, list)) else y_true
        
        # get predictions from the network and measure inference time
        pred, std = uncertainty_propagator.predict(imgs)
        
        # if pred is a tuple or list, take the first network output
        y_pred = pred[0] if isinstance(pred, (tuple, list)) else pred
        # extract the label using ArgMax and flatten into a 1D vector
        y_true = np.argmax(y_true, axis=-1).flatten()
        y_pred = np.argmax(y_pred, axis=-1).flatten()
        # calculate the confusion matrix and add to the accumulated matrix
        confusion += confusion_matrix(y_true, y_pred, list(range(num_classes)))
        
        # save uncertainty values with correct/incorrect prediction
        unc_pred_list.append([std.flatten(), y_true==y_pred])
        
        # save uncertainty values depending on class
        for i in range(unc_of_class.shape[0]):
            mask_i = y_true==i
            unc_of_class[i] += std.flatten()[mask_i].sum()
            class_counts[i] += std.flatten()[mask_i].size
        
        if len(class_holdout) > 0:
            # create class hold mask
            holdout_mask = np.zeros(y_true.shape)
            for ch in class_holdout:
                holdout_mask += y_true == ch
            holdout_mask = holdout_mask > 0
            # append stds
            unc_holdout.append(std.flatten()[holdout_mask])
            unc_not_holdout.append(std.flatten()[np.logical_not(holdout_mask)])
    
    if len(class_holdout) > 0:
        # determine mean uncertainty on hold out and not hold out
        unc_holdout_mean = np.concatenate(unc_holdout, axis=0).mean()
        unc_not_holdout_mean = np.concatenate(unc_not_holdout, axis=0).mean()
    else: 
        unc_holdout_mean = None
        unc_not_holdout_mean = None
        
    # calculate the metrics from the predicted and ground truth values
    accuracy, mean_per_class_accuracy, per_class_accuracy, mean_iou, iou = metrics(confusion, mask)
    # build a dictionary to store metrics in
    _metrics = {
        '#1#Accuracy': accuracy,
        '#2#Mean Per Class Accuracy': mean_per_class_accuracy,
        '#3#Mean I/U': mean_iou,
    }
    # set the label map to an empty dictionary if it's None
    code_map = code_map if code_map is not None else dict()
    # iterate over the labels and I/Us in the vector of I/Us
    for label, iou_c in enumerate(iou):
        # if the value is in the mask, add it's value to the metrics dictionary
        if mask[label]:
            _metrics[code_map.get(label, str(label)) + '_IOU'] = iou_c
    for label, acc_c in enumerate(per_class_accuracy):
        # if the value is in the mask, add it's value to the metrics dictionary
        if mask[label]:
            _metrics[code_map.get(label, str(label)) + '_ACC'] = acc_c
    # create a series with the metrics data
    _metrics = pd.Series(_metrics).sort_index()
    # replace the markers that force the core metrics to the top
    _metrics.index = _metrics.index.str.replace(r'#\d#', '')
    # convert the series to a DataFrame before returning
    
    # get data for callibration plot
    callibration = get_uncertainty_callibration(unc_pred_list)
    
    return pd.DataFrame(_metrics, columns=['Value']), callibration, [unc_holdout_mean, unc_not_holdout_mean], unc_of_class/class_counts
    

# explicitly define the outward facing API of this module
__all__ = [evaluate.__name__]
