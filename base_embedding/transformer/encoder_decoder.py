from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Layer
from .attention import MultiHeadAttention


class EncoderLayer(Layer):

    def __init__(self, head_nums, d_model, dff, rate):
        super(EncoderLayer, self).__init__()

        self.d_model = d_model
        self.head_nums = head_nums
        self.dff = dff
        self.rate = rate

        self.mha = MultiHeadAttention(head_nums, d_model)

        # define Point wise feed forward network
        self.ffn1 = Dense(dff)
        self.fnn2 = Dense(d_model, activation="relu")

        # define LayerNormalization Layer
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        # define dropout layer
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    # def build(self, input_shape):
    #     return input_shape

    def call(self, inputs, *args, **kwargs):
        attn_output = self.mha(inputs)
        attn_output = self.dropout1(attn_output)
        output_norm1 = self.layernorm1(inputs+attn_output)

        ffn1 = self.ffn1(output_norm1)
        ffn2 = self.ffn2(ffn1)
        ffn2 = self.dropout2(ffn2)
        ffn_output = self.layernorm2(ffn2 + output_norm1)
        return ffn_output

    def compute_output_shape(self, input_shape):
        return input_shape


class DecoderLayer(Layer):

    def __init__(self, head_nums, d_model, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.head_nums = head_nums
        self.d_model = d_model
        self.dff = dff
        self.rate = rate

        # define dropout layer
        self.mha1 = MultiHeadAttention(head_nums, d_model)
        self.mha2 = MultiHeadAttention(head_nums, d_model)

        # point wise feed forward network
        self.ffn1 = Dense(dff)
        self.ffn2 = Dense(d_model)

        # define normalization layer
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)

        # define dropout layer
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.dropout3 = Dropout(rate)

    def build(self, input_shape):
        return input_shape

    def call(self, inputs, *args, **kwargs):
        attn_output1 = self.mha1(inputs)
        dropout1 = self.dropout1(attn_output1)
        norm_output1 = self.layernorm1(inputs+dropout1)

        attn_output2 = self.mha2(norm_output1)
        dropout2 = self.dropout2(attn_output2)
        norm_output2 = self.layernorm2(norm_output1+dropout2)

        ffn_output1 = self.ffn1(norm_output2)
        ffn_output2 = self.ffn2(ffn_output1)
        dropout3 = self.dropout3(ffn_output2)
        norm_output3 = self.layernorm3(norm_output2+dropout3)

        return norm_output3

    def compute_output_shape(self, input_shape):
        return input_shape
