
import tensorflow as tf
from base_embedding.transformer.encoder_decoder import EncoderLayer, DecoderLayer


encode_layer = EncoderLayer(8, 512, 2048)
input = tf.random.uniform((64, 43, 512))
output = encode_layer(input)

print(output.shape)
