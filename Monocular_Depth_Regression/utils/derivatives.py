import tensorflow as tf


def dElu_tf(x):
    smaller_zero_mask = tf.to_float(x < 0)
    greater_equal_zero_mask = tf.to_float(x >= 0)
    return greater_equal_zero_mask + tf.exp(x) * smaller_zero_mask

def dSigmoid_tf(x):
    return tf.nn.sigmoid(x) * (1 - tf.nn.sigmoid(x))