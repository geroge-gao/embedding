import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import GlorotUniform




class Self_Attention(Layer):

    def __init__(self, output_dim, name, mask):
        self.name = name
        self.output_dim = output_dim

    
    def build(self, input_shape):
        self.kernel = self.add_weight(name="kernel",
                                    shape=(3, input_shape[-1], self.output_dim),
                                    initializer=GlorotUniform,
                                    trainable=True)
        return super(Self_Attention, self).build(input_shape)

    
    def call(self, x, mask=None):
        """ 
            x: raw input, [batch_size, seq_len, input_dim]
            k: weight, [1, input_dim, output_dim]
        """

        # get KQV
        k = tf.matmul(x, self.kernel[0])
        q = tf.batch_dot(x, self.kernel[1]) 
        v = tf.batch_dot(x, self.kernel[2])

        # calculate dk
        dk = tf.sqrt(self.output_dim)

        k_t = tf.transpose(k, perm=[0, 2, 1])
        qk = tf.matmul(q, k_t)
        # scaled_attention_weighhts = [1, sequence_k, sequence_v]
        # TODO(gerogegao): add mask function
        scaled_attention_weights = tf.nn.softmax(qk/dk, axis=-1)

        # add mask
        if mask is not None:
            # 将mask转换成int类
            mask = tf.cast(mask, tf.int32)
            mask = tf.not_equal(mask, 0)
            scaled_attention_weights += mask * -1e9
        
        output = tf.matmul(scaled_attention_weights, v)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_shape)


class MultiHeadAttention(Layer):

    def __init__(self, ):
        pass


    def call(self, inputs, *args, **kwargs):
        return super().call(inputs, *args, **kwargs)