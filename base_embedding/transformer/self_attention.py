from tensorflow.keras.layers import Layer


class Self_Attention(Layer):
    """
        self attention layer 
    """

    def __init__(self):
        self.output_shape


    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x):
        # x: [batch_size, sequence_length, embedding_size]
        pass


    def compute_output_shape(self):
        return self.output_shape


    