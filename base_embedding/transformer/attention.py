import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense, Dropout, Concatenate
from tensorflow.keras.initializers import GlorotUniform


class SelfAttention(Layer):

    def __init__(self, output_dim):
        self.output_dim = output_dim
        self.kernel = None

        self.k = Dense(output_dim)
        self.q = Dense(output_dim)
        self.v = Dense(output_dim)
    
    def build(self, input_shape):
        # input_shape: [batch_size, head_nums, seq_len, dk]
        # kernel_shape: [3, head_nums, seq_len, dk]
        self.kernel = self.add_weight(name="kernel",
                                      shape=(3, input_shape[1], input_shape[2], input_shape[2]),
                                      initializer=GlorotUniform,
                                      trainable=True)
        return super(SelfAttention, self).build(input_shape)
    
    def call(self, x, mask=None):
        """ 
            x: raw input, [batch_size, head_nums ,seq_len, dk]
            k: weight, [1, head_nums, seq_len, d_k]
        """

        # get KQV [batch_size, head_nums, seq_len, d_model]
        k = self.k(x)
        q = self.q(x)
        v = self.v(x)

        # calculate qk
        # qk: [batch_size, head_nums, seq_len_q, seq_len_k]
        qk = tf.matmul(q, k, transpose_b=True)
        scaled_attention_weights = qk / self.dk

        # add mask
        if mask is not None:
            # 将mask转换成int类
            mask = tf.cast(mask, tf.int32)
            mask = tf.not_equal(mask, 0)
            scaled_attention_weights += mask * -1e9
        # scaled_attention_weights: [ ,head_nums, seq_len_q, seq_len_v]
        scaled_attention_weights = tf.nn.softmax(scaled_attention_weights, axis=-1)
        # output: [batch_size, head_nums, seq_len, d_model]
        output = tf.matmul(scaled_attention_weights, v)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim


class MultiHeadAttention(Layer):

    def __init__(self, multi_head_nums, output_dim):
        self.multi_head_nums = multi_head_nums
        self.dk = output_dim // multi_head_nums
        self.output_dim = output_dim
        self.attention = SelfAttention(output_dim)

        self.dense = Dense(output_dim)

    def call(self, inputs, batch_size):
        multi_attentions = []
        # input shape: [batch_size, seq_len, d_model]
        inputs = tf.reshape(inputs, (batch_size, -1, self.multi_head_nums, self.dk))
        # x: [batch_size, head_nums, seq_len, dk]
        x = tf.transpose(inputs, perm=[0, 2, 1, 3])
        output = self.attention(x)
        return output

