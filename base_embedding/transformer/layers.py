# #!/usr/bin/python
# -*- coding: UTF-8 -*-

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Concatenate
from attention import Self_Attention

class FFNLayer(Layer):

    def __init__(self,):
        pass

    
    def build(self, input_shape):
        return super().build(input_shape)


    def call(self, input_shape):
        pass

    def compute_output_shape():
        pass



class EncoderLayer(Layer):

    def __init__(self, d_model, num_heads, dff, mask_zero, rate=0.1, ):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate
        self.mask = mask_zero
        self.attentions = []
        for i in range(num_heads):
            self_attention = Self_Attention(self.d_model, name="head_{}".format(i), mask=self.mask)
            self.attentions.append(self_attention)

        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, mask):
        pass

    
    def compute_mask(self, inputs, mask=None):
        if self.mask:
            return tf.not_equal(inputs, 0)
        else:
            return None 


class DecoderLayer(Layer):

    def __init__(self,):
        pass