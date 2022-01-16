import tensorflow as tf

from base_embedding.transformer.encoder_decoder import EncoderLayer, DecoderLayer
from base_embedding.transformer.attention import MultiHeadAttention

# encode_layer = EncoderLayer(8, 512, 2048)
# inputs = tf.random.uniform((64, 43, 512))
# output = encode_layer(inputs)
#
# print(output.shape)
d_model = 512
num_heads = 8
mha = MultiHeadAttention(num_heads, d_model)
y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
output = mha(y)
print(output.shape)
