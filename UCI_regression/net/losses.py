import math
import tensorflow as tf
from tensorflow.python import assert_non_negative


def mdn_loss_wrapper(modes: int=1):

    def mdn_loss(y_true, y_pred):
        """
        Computes mdn loss
        Fixed to 1 mode!

        args:
            y_true: labels plus dummy labels for std and weight
            y_pred: prediction

        returns:
            loss
        """

        targets = y_true[:, :1]
        means = y_pred[:, :modes]

        var = y_pred[:, modes:2 * modes]
        var = tf.debugging.check_numerics(var, 'var 01')

        modes_prob = []
        for i in range(modes):
            diff = tf.subtract(targets, means[:, i:(i + 1)])
            diff = tf.debugging.check_numerics(diff, 'diff 01')
            m_loss_log = -1 * tf.divide(tf.square(diff), 2 * var[:, i:(i + 1)]) - \
                         tf.log(tf.sqrt(2 * math.pi * var[:, i:(i + 1)]))  # + tf.log(weights[:, i:(i + 1)])
            m_loss_log = tf.debugging.check_numerics(m_loss_log, 'm_loss_log 01')
            modes_prob.append(m_loss_log)

        prob = tf.concat(modes_prob, axis=1)

        loss = tf.reduce_logsumexp(prob, axis=1)
        loss = tf.debugging.check_numerics(loss, 'loss 01')
        loss = tf.reduce_mean(loss, axis=0)

        return (-1) * loss

    return mdn_loss


def learn_dropout_loss(y_true, y_pred):
    """
    Computes loss for learning dropout rate
    Similar to MDN loss but with only one mode and no weights
    Further does this require the direct predict of the variance
    and not its logarithm

    args:
        y_true: labels plus dummy labels for std
        y_pred: prediction

    returns:
        loss
    """

    targets = y_true[:, :1]
    means = y_pred[:, :1]
    var = y_pred[:, 1:2]

    var = assert_non_negative_and_not_nan(var, 'Var in loss')

    diff = tf.subtract(targets, means)
    diff = tf.debugging.check_numerics(diff, 'diff 01')
    loss_log = tf.divide(tf.square(diff), 2 * var + 1e-8) + tf.log(tf.sqrt(2 * math.pi * var + 1e-8))
    loss_log = tf.debugging.check_numerics(loss_log, 'loss_log 01')

    loss = tf.reduce_mean(loss_log, axis=0)

    return loss


def assert_non_negative_and_not_nan(x, msg):
    assert_op = assert_non_negative(x, message=msg)
    with tf.control_dependencies([assert_op]):
        x = tf.debugging.check_numerics(x, msg)
    return x
