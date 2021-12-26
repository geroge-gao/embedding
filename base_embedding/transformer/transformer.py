#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""

"""
 
import pandas as pd
import tensorflow as tf

from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.layers import Layer



class transformer(Layer):

    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable=trainable, name=name, dtype=dtype, dynamic=dynamic, **kwargs)

