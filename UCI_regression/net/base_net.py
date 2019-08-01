import os
import datetime
import numpy as np
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping


class BaseNet:

    def __init__(self, debug: bool=False):
        self.debug = debug

    def normalize_data(self, X_train, y_train, X_val, y_val):
        self.std_X_train = np.std(X_train, 0)
        self.std_X_train[self.std_X_train == 0] = 1
        self.mean_X_train = np.mean(X_train, 0)

        self.X_train = (X_train - np.full(X_train.shape, self.mean_X_train)) / \
                       np.full(X_train.shape, self.std_X_train)
        self.X_val = (X_val - np.full(X_val.shape, self.mean_X_train)) / \
                       np.full(X_val.shape, self.std_X_train)

        self.mean_y_train = np.mean(y_train)
        self.std_y_train = np.std(y_train)

        self.y_train_normalized = (y_train - self.mean_y_train) / self.std_y_train
        self.y_train_normalized = np.array(self.y_train_normalized, ndmin=2).T

        self.y_val_normalized = (y_val - self.mean_y_train) / self.std_y_train
        self.y_val_normalized = np.array(self.y_val_normalized, ndmin=2).T

    def get_callbacks(self):

        callbacks = []

        # callbacks.append(EarlyStopping(patience=10))
        if self.debug:
            log_dir = 'tensorboard_logs/' + str(datetime.datetime.now())
            os.mkdir(log_dir)
            callbacks.append(TensorBoard(log_dir=log_dir, histogram_freq=1, write_grads=True))

        return callbacks
