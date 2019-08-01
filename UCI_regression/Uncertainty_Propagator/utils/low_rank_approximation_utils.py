import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

from ..utils.helper import red_string_matmul


def get_cov_left_right(cov_diag, k):
    
    # shape_cov = cov_diag.get_shape().as_list()
    shape_cov = tf.shape(cov_diag)
    
    # create index mask for largest k variances
    _, top_idx = tf.nn.top_k(cov_diag, k=k)
    ii, _ = tf.meshgrid(
        tf.range(shape_cov[0]),
        tf.range(k),
        indexing='ij'
    )
    top_idx = tf.stack([ii, top_idx], axis=-1)
    
    # create reduced covariance
    top_k = tf.gather_nd(cov_diag, top_idx)
    top_k_cov = tf.linalg.diag(top_k)
    
    # create batch of unit matrices and
    # select rows corresponding to highest variance
    eye = tf.eye(shape_cov[-1], batch_shape=[shape_cov[0]])
    eye_top_rows = tf.gather_nd(eye, top_idx)
    
    # left and right covariances matrix Sigma = U L V^T
    # by convention U and L are combined
    shape_eye = eye_top_rows.get_shape().as_list()
    red_string = red_string_matmul(eye_top_rows, top_k_cov)
    permutation = list(np.arange(len(shape_eye)))
    permutation = permutation[:-2] + permutation[-1:] + permutation[-2:-1]
    cov_left = tf.einsum(red_string,
                         tf.transpose(eye_top_rows, perm=permutation), 
                         top_k_cov)
    cov_right = eye_top_rows
    
    return [cov_left, cov_right]


def reconst_covariance(l_cov, r_cov):
    red_string = red_string_matmul(l_cov, r_cov)
    return tf.einsum(red_string, l_cov, r_cov)


def mult_lr_cov(lr_cov, J):

    l, r = lr_cov
    shape_j = J.get_shape().as_list()
    red_string1 = red_string_matmul(J, l)
    red_string2 = red_string_matmul(r, J)

    # propagate left cov
    permutation = list(np.arange(len(shape_j)))
    permutation = permutation[:-2] + permutation[-1:] + permutation[-2:-1]
    l = tf.einsum(red_string1, tf.transpose(J, perm=permutation), l)

    # propagation right cov
    r = tf.einsum(red_string2, r, J)

    return [l, r]


def merge_lr_cov(l, r):
    red_string = red_string_matmul(l, r)
    return tf.einsum(red_string, l, r)


class MergeLRPropagationLayer(Layer):
    """
    Layer for merging left and right part of covariance matrix
    """

    def __init__(self, **kwargs):
        super(MergeLRPropagationLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MergeLRPropagationLayer, self).build(input_shape)

    def call(self, x, **kwargs):
        """
        call method using full covariance or diagonal covariance
        """
        return merge_lr_cov(x[0], x[1])

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)
