import tensorflow as tf
from tensorflow.keras.layers import Layer

from tensorflow.keras import backend as K



class Self_Attention(Layer):

    def __init__(self, output_dim, name):
        self.name = name
        self.output_dim = output_dim

    
    def build(self, input_shape):
        self.kernel = self.add_weight(name="kernel",
                                    shape=(3, input_shape[-1], self.output_dim),
                                    initializer="uniform",
                                    trainable=True)
        return super(Self_Attention, self).build(input_shape)

    
    def call(self, x):
        """ 
            x: raw input, [batch_size, seq_len, input_dim]
            k: weight, [input_dim, output_dim]
        """
        k = K.batch_dot(x, self.kernel[0])
        q = K.batch_dot(x, self.kernel[1]) 
        v = K.batch_dot(x, self.kernel[2])

        dk = k.shape[-1]






    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_shape)


        

class MultiHeadAttention(Layer):

    def __init__(self):
        pass