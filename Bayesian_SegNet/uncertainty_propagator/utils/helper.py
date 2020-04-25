import math
import numpy as np
import tensorflow as tf


def batch_std(x:np.array, batch_axis: int=0) -> np.array:
    assert len(x.shape) >= batch_axis
    return np.std(x, axis=batch_axis)

def relu_var_tf(mean, var, eps=1e-8):
    std = tf.sqrt(var+eps)
    exp = mean/(tf.sqrt(2.0)*std)
    erf_exp = tf.math.erf(exp)
    exp_exp2 = tf.exp(-1*exp**2)
    term1 = 0.5 * (var+mean**2) * (erf_exp + 1)
    term2 = mean*std/(tf.sqrt(2*math.pi))*exp_exp2
    term3 = mean/2*(1+erf_exp)
    term4 = tf.sqrt(1/2/math.pi)*std*exp_exp2
    return tf.nn.relu(term1 + term2 - (term3 + term4)**2)

def variance_product_rnd_vars(mean1, mean2, var1, var2):
    return mean1**2*var2 + mean2**2*var1 + var1*var2

def d_softmax_tf(var, sm):
    sm_shape = sm.get_shape().as_list()
    dims = len(sm_shape)-2
    unit = tf.eye(sm_shape[-1], batch_shape=tf.shape(sm)[:-1])
    diff = tf.subtract(unit, tf.expand_dims(sm, -2))
    red_str1, red_str2 = build_einsum_red_string(dims)
    d_sm_tf = tf.einsum(red_str1, diff, sm)
    out_var = tf.einsum(red_str2, d_sm_tf**2, var)
    return out_var

def build_einsum_red_string(dims: int=0):
    chars = ['i', 'j', 'k', 'l', 'm', 'n', 'o', 'p']
    assert dims < len(chars)-3 and dims >= 0
    base_str = ''.join([s for s in chars[:dims+3]])
    red_str1 = base_str + ',' + base_str[:-1]  + '->' + base_str
    red_str2 = base_str + ',' + base_str[:-2] + base_str[-1]  + '->' + base_str[:-1]
    return red_str1, red_str2