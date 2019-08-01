"""A Keras callback to plot metrics during fitting.

Derived from: https://github.com/stared/livelossplot

"""
import matplotlib.pyplot as plt
from IPython import display
from keras.callbacks import Callback


def draw_plot(logs: list, metrics: list, max_epoch: int,
    figsize: tuple=None,
    max_cols: int=2,
    validation_fmt: str="val_{}",
):
    """
    Plot Keras metrics data.

    Args:
        logs: the logs from the fit call
        metrics: the list of metrics to plot
        max_epoch: the max epoch for the training operation
        figsize: the size of the figure if defined
        max_cols: the max columns for the plot
        validation_fmt: the format string for validation metrics

    Returns:
        None

    """
    # clear the current output display
    display.clear_output(wait=True)
    # setup a new figure
    plt.figure(figsize=figsize)
    # iterate over the list of metrics
    for metric_id, metric in enumerate(metrics):
        plt.subplot((len(metrics) + 1) / max_cols + 1, max_cols, metric_id + 1)
        plt.xlim(1, max_epoch)
        train_metric = [log[metric] for log in logs]
        plt.plot(range(1, len(logs) + 1), train_metric, label="training")

        if validation_fmt.format(metric) in logs[0]:
            val_metric = [log[validation_fmt.format(metric)] for log in logs]
            plt.plot(range(1, len(logs) + 1), val_metric, label="validation")

        plt.title(metric)
        plt.xlabel('epoch')
        plt.legend(loc='center right')

    plt.tight_layout()
    plt.show()


class PlotMetrics(Callback):
    """A Keras callback to plot metrics during fitting."""

    def __init__(self,
        figsize: tuple=None,
        cell_size: tuple=(6, 4),
        max_cols: int=2
    ):
        """
        Initialize a new Keras Metrics plot callback.

        Args:
            figsize: the size of the figure
            cell_size: the size of each subplot
            max_cols: the max number of columns in the plot

        Returns:
            None

        """
        self._figsize = figsize
        self._cell_size = cell_size
        self._max_cols = max_cols
        self._metrics = None
        self._max_epoch = None
        self._logs = None

    def on_train_begin(self, logs: dict={}):
        """
        Setup the callback at the beginning of training.

        Args:
            logs: the logs from the training process

        Returns:
            None

        """
        # setup the list of metrics
        self._metrics = []
        for metric in self.params['metrics']:
            if metric.startswith('val_'):
                continue
            self._metrics.append(metric)
        # setup the fig size if not defined
        if self._figsize is None:
            self._figsize = (
                self._max_cols * self._cell_size[0],
                ((len(self._metrics) + 1) / self._max_cols + 1) * self._cell_size[1]
            )
        # get the max number of epochs
        self._max_epoch = self.params['epochs']
        # set the logs to an empty list
        self._logs = []

    def on_epoch_end(self, epoch: int, logs: dict={}):
        """
        Handle the end of an epoch by plotting the log data.

        Args:
            epoch: the epoch to plot data for
            logs: the logs from the last epoch

        Returns:
            None

        """
        # add the logs to the list
        self._logs.append(logs.copy())
        # plot the metrics data for the history of logs
        draw_plot(self._logs, self._metrics,
            figsize=self._figsize,
            max_epoch=self._max_epoch,
            max_cols=self._max_cols,
            validation_fmt="val_{}",
        )


# explicitly define the outward facing API of the module
__all__ = [PlotMetrics.__name__]
