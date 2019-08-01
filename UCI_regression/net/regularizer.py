import tensorflow as tf
from tensorflow.keras.regularizers import L1L2


class GaussianPriorRegularizer(L1L2):

    def __init__(self, factor, l: float=0.01, e: float=0.01, **kwargs):
        self.factor = factor
        self.e = e
        super(GaussianPriorRegularizer, self).__init__(l2=l, **kwargs)

    def __call__(self, x):
        x = self.factor * x
        weight_reg_result = super(GaussianPriorRegularizer, self).__call__(x)
        return weight_reg_result  - self.e* (tf.log(self.factor)*self.factor + (1-self.factor)*tf.log(self.factor))
