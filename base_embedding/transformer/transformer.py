
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding, Dropout
from tensorflow.keras import Input, Model
from .encoder_decoder import EncoderLayer, DecoderLayer
from .utils import positional_encoding


def transformer(d_model, layer_nums, head_nums, dff, input_size, target_size, pe_input, pe_target, rate=0.1):
    """
        transformer model, include n encoder layer and n decoder layer
    Parameters:
        d_model: int, model output dim
        layer_nums: int, layer number of EncoderLayer and DecoderLayer
        head_nums: int, numbers of self attention in one coder layer
        dff: int, the dim of point wise feed forward network's first layer
        input_size: int, input vocabulary word numbers
        target_size: int, output vocabulary word numbers
        pe_input: int, the max number of input sequence's position encoding
        pe_target: int ,the max number of target sequence's position encoding
        rate: double, dropout rate

    Returns:
    """

    x = Input(pe_input, )
    embed = Embedding(input_dim=input_size, output_dim=d_model)(x) # [batch_size, sequence, dim_output]
    # add position encoding
    pos_encoding = positional_encoding(pe_input, d_model)
    embed *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embed += pos_encoding
    encoder_input = Dropout(rate)(embed)

    # encoder parts
    for i in range(layer_nums):
        encoder_input = EncoderLayer(head_nums, d_model, head_nums, rate)(encoder_input)
    encoder_output = Dropout(encoder_input)

    # decoder parts
    decoder_input = Embedding(target_size, d_model)(encoder_output)
    decoder_input *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    pe_target_encoding = positional_encoding(pe_target, d_model)
    decoder_input += pe_target_encoding
    decoder_input = Dropout(rate)(decoder_input)
    for i in range(layer_nums):
        decoder_input = DecoderLayer(head_nums, d_model, dff, rate)(decoder_input)

    # final output layer
    y = Dense(target_size, )(decoder_input)
    model = Model(inputs=x, outputs=y)
    return model






