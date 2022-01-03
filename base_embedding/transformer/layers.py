# #!/usr/bin/python
# -*- coding: UTF-8 -*-

from tensorflow.keras.layers import Layer


class EncoderLayer(Layer):

    def __init__(self, name):
        self.name = name

    
    def build(self, input_shape):
        return super().build(input_shape)


    def call(self, input_shape):
        pass

    def compute_output_shape():
        pass