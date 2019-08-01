import math
import numpy as np
import tensorflow as tf


def batch_std(x:np.array, batch_axis: int=0) -> np.array:
    assert len(x.shape) >= batch_axis
    return np.std(x, axis=batch_axis)


def relu_var_tf(mean, var, eps=1e-8):
    std = tf.sqrt(var+eps)
    exp = mean/(tf.sqrt(2.0)*std)
    erf_exp = tf.erf(exp)
    exp_exp2 = tf.exp(-1*exp**2)
    term1 = 0.5 * (var+mean**2) * (erf_exp + 1)
    term2 = mean*std/(tf.sqrt(2*math.pi))*exp_exp2
    term3 = mean/2*(1+erf_exp)
    term4 = tf.sqrt(1/2/math.pi)*std*exp_exp2
    return tf.nn.relu(term1 + term2 - (term3 + term4)**2)


def jacobian_tanh_tf(x):
    """only diagonal of J needed bc tanh is applied element-wise
    
    args:
        x: tensor, input to the tanh activation function
    """
    return 1 - tf.nn.tanh(x)**2


def variance_product_rnd_vars(mean1, mean2, var1, var2):
    return mean1**2*var2 + mean2**2*var1 + var1*var2


def covariance_elementwise_product_rnd_vec(mean1, cov1, mean2, var2):
    """
    Computes covariance element-wise product of one vector (->1)
    with full covariance and another vector with independent elements
    """

    var2_scalar = var2.get_shape().as_list()[0] == 1

    # term 1
    var1 = tf.matrix_diag_part(cov1)
    if var2_scalar:
        term1 = var1 * var2
    else:
        term1 = tf.einsum('ij,j->ij', var1, var2)
    term1 = tf.matrix_diag(term1)

    # term 2
    if var2_scalar:
        term2 = mean1 ** 2 * var2
    else:
        term2 = tf.einsum('ij,j->ij', mean1**2, var2)
    term2 = tf.matrix_diag(term2)

    # term 3
    if var2_scalar:
        term3 = mean2 ** 2 * cov1
    else:
        mean2_cross = tf.einsum('i,j->ij', mean2, mean2)
        term3 = tf.einsum('ijk,jk->ijk', cov1, mean2_cross)

    return term1 + term2 + term3


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


def red_string_matmul(t1: tf.Tensor, t2:tf.Tensor):
    dim1 = len(t1.get_shape().as_list())
    dim2 = len(t2.get_shape().as_list())
    diff= dim1 - dim2
    assert dim1 >= 2 and dim2 >= 2
    chars = ['i', 'j', 'k', 'l', 'm', 'n', 'o', 'p']
    str1 = ''.join(chars[:dim1])
    if diff >= 0:
        str2 = ''.join(chars[:dim2-2]+[chars[dim1-1]]+[chars[dim1]])
        str3 = ''.join(chars[:dim1-1]+[chars[dim1]])
    else:
        str2 = ''.join(chars[:dim1-2]+chars[dim1:dim1-diff]+chars[dim1-1:dim1]
                       +chars[dim1-diff:dim1-diff+1])
        str3 = str2[:-2] + str1[-2] + str2[-1]
    return str1 + ',' + str2 + '->' + str3


def tf_gather_batch(batch, idx):
    batch_unrolled = tf.reshape(batch, shape=[-1])
    idx_unrolled = tf.reshape(idx, shape=[-1])
    batch_gathered = tf.gather(batch_unrolled, idx_unrolled)
    batch_gathered = tf.reshape(batch_gathered, shape=tf.shape(idx))
    return batch_gathered


def jacobian_ReLU_tf(x):
    mask_int = tf.to_float(x > 0)
    return tf.linalg.diag(mask_int)
