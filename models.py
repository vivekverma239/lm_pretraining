import os
import json
import pickle

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers

def lookup_layer(item_to_lookup, vocab_file, num_oov_buckets=1):
    vocab = tf.contrib.lookup.index_table_from_file(
                        vocab_file, num_oov_buckets=num_oov_buckets)
    return vocab.lookup(item_to_lookup)
#
#
# def PretrainedLSTM(save_path, input_layer=None, return_sequences=False):
#     """
#
#         :params:
#             - save_path : Folder where pretrained files have been saved
#             - input_layer (optional): Should be keras Input layer with tf.string input
#
#         :output:
#             - keras model
#
#         NOTE: you have to call tf.tables_initializer().run() somewhere before fitting
#         the data.
#
#     """
#     assert  os.path.exists(save_path), "{} doesn't exist".format(save_path)
#     config = json.load(open(os.path.join(save_path, "config.json")))
#     weights = pickle.load(open(os.path.join(save_path, "weights.pkl"), "rb"))
#     vocab_file = os.path.join(save_path, "vocab.txt")
#
#     #config
#     max_vocab_size = config["max_vocab_size"]
#     embed_size = config["embed_size"]
#     hidden_size = config["hidden_size"]
#
#     if input_layer ==None:
#         input_layer = layers.Input(shape=(None,), dtype="string")
#
#     lookup_vocab = layers.Lambda(lambda x: lookup_layer(x, vocab_file))
#     input_layer_idx = lookup_vocab(input_layer)
#
#     embeded_input = layers.Embedding(max_vocab_size, embed_size,
#                                      weights=weights["embedding"])(input_layer_idx)
#     embeded_input = layers.Dropout(0.5)(embeded_input)
#     rnn_output = layers.CuDNNLSTM(units=hidden_size,
#                                   return_sequences=return_sequences,
#                                   weights=weights["lstm"])(embeded_input)
#
#     model = Model(inputs=[input_layer], outputs=[rnn_output])
#     return model

def PretrainedLSTM(save_path, input_layer=None, return_sequences=False, load_weights=True):
    """

        :params:
            - save_path : Folder where pretrained files have been saved
            - input_layer (optional): Should be keras Input layer with tf.string input

        :output:
            - keras model

        NOTE: you have to call tf.tables_initializer().run() somewhere before fitting
        the data.

    """
    assert  os.path.exists(save_path), "{} doesn't exist".format(save_path)
    config = json.load(open(os.path.join(save_path, "config.json")))
    weights = pickle.load(open(os.path.join(save_path, "weights.pkl"), "rb"))
    vocab_file = os.path.join(save_path, "vocab.txt")

    #config
    max_vocab_size = config["max_vocab_size"]
    embed_size = config["embed_size"]
    hidden_size = config["hidden_size"]
    num_layers = config["num_layers"]

    if input_layer ==None:
        input_layer = layers.Input(shape=(None,), dtype="string")

    lookup_vocab = layers.Lambda(lambda x: lookup_layer(x, vocab_file))
    input_layer_idx = lookup_vocab(input_layer)

    embeded_input = layers.Embedding(max_vocab_size, embed_size,
                                     weights=weights["embedding"] if load_weights else None)\
                                     (input_layer_idx)
    embeded_input = layers.Dropout(0.3)(embeded_input)
    rnn_input = embeded_input
    for i in range(num_layers):
        rnn_output = layers.CuDNNLSTM(units=hidden_size,
                                      return_sequences=return_sequences,
                                      weights=weights["lstm"][i] if load_weights
                                                            else None)(rnn_input)

    model = Model(inputs=[input_layer], outputs=[rnn_output])
    return model


if __name__ == '__main__':
    PretrainedLSTM(save_path="saved_model/base")
