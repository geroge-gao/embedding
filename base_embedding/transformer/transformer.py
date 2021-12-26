#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""

"""
 
import pandas as pd
import tensorflow as tf

from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.layers import Layer





class Self_Attention(Layer):

    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)

    def build(self, input_shape):
        return super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        return super().call(inputs, *args, **kwargs)

    def compute_output_shape(self, input_shape):
        return super().compute_output_shape(input_shape)




class transformer(Layer):

    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)


    def build(self, input_shape):
        return super().build(input_shape)


    def call(self, inputs, *args, **kwargs):
        return super().call(inputs, *args, **kwargs)

    
    def compute_output_shape(self, input_shape):
        return super().compute_output_shape(input_shape)

