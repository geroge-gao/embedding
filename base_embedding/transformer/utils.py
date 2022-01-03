import numpy as np
import tensorflow as tf


def get_angles(pos, i, d_model=None):
    angles_rate = np.power(10000, 2 * (i // 2) * d_model)
    return pos / angles_rate 


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # get odd rows
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # get even rows
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    # add dimensions to angle
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


def padding_mask():
    pass # TOD


def sequence_mask():
    pass # TOD


def scaled_dot_product_attention(q, k, v, mask):

    
