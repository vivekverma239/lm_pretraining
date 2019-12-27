import os
import fire
from tqdm import tqdm
import random
import tensorflow as tf
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import random
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.preprocessing import text, sequence

from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, Embedding, Concatenate, \
                                     Bidirectional, Dense, \
                                    GlobalAveragePooling1D, GlobalMaxPooling1D,\
                                    Conv1D, LSTM, Add, BatchNormalization,\
                                    Activation, Dropout, Reshape,\
                                    Conv2D, MaxPooling2D, Flatten, Subtract, \
                                    Softmax, Dot, TimeDistributed, Lambda

from tensorflow.compat.v1.keras.layers import CuDNNLSTM, CuDNNGRU

from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from models import PretrainedLSTM
from data_utils import get_tokenizer
from examples.attention import Attention, Combine
# python -m data.imdb --save_path data/imdb --imdb_path data/imdb/aclImdb/
TEST_POS_DIR = 'test/pos'
TEST_NEG_DIR = 'test/neg'
TRAIN_POS_DIR = 'train/pos/'
TRAIN_NEG_DIR = 'train/neg/'




def _load_quora_data(data_file,\
                    max_length=60,
                    validation_split=5000,
                    test_split=5000,
                    seed=100,
                    tokenizer="nltk",
                    processor_config_filepath='preprocessor.pkl'):
    """
        Load Quora Dataset from TSV file
        :params:
            - data_file: TSV data file provided by Quora
            - max_length: Max Length of Questions, Questions data will be
                                truncated upto this length
            - validation_split: How much to sample for validation
            - test_split: How much to sample for testing
            - seed: Random seed
            - processor_config_filepath: Where to save tokenizer etc 
    """
    # Read data file and assign column names
    data = pd.read_csv(data_file, sep='\t')

    # Shuffle and split dataframe
    np.random.seed(seed)
    data.iloc[np.random.permutation(len(data))]

    train_df, valid_df, test_df = data.iloc[:-(validation_split+test_split)],\
                                  data.iloc[-(validation_split+test_split):-test_split],\
                                  data.iloc[-test_split:, :]

    convert_list_to_str = lambda x: list(map(str,x))
    train_question1 = convert_list_to_str(train_df['question1'].tolist())
    train_question2 = convert_list_to_str(train_df['question2'].tolist())
    y_train = train_df['is_duplicate']

    valid_question1 = convert_list_to_str(valid_df['question1'].tolist())
    valid_question2 = convert_list_to_str(valid_df['question2'].tolist())
    y_valid = valid_df['is_duplicate']

    test_question1 = convert_list_to_str(test_df['question1'].tolist())
    test_question2 = convert_list_to_str(test_df['question2'].tolist())
    y_test = test_df['is_duplicate']

    if not tokenizer:
        tokenizer = text_to_word_sequence
    else:
        tokenizer = get_tokenizer(get_tokenizer)

    def process_list_of_text(list_of_text):
        tokenized = list(map(tokenizer, x_train))
        return pad_sequences(tokenized, maxlen=max_length, dtype=object, padding='pre', value="-pad-")



    # Processing Training Data
    train_question1 = process_list_of_text(train_question1)
    train_question2 = process_list_of_text(train_question2)

    # Processing Validation Data
    valid_question1 = process_list_of_text(valid_question1)
    valid_question2 = process_list_of_text(valid_question2)

    # Processing Test Data
    test_question1 = process_list_of_text(test_question1)
    test_question2 = process_list_of_text(test_question2)


    return  (train_question1, train_question2), y_train,\
                        (valid_question1, valid_question2), y_valid,\
                        (test_question1, test_question2), y_test


def get_model_v4(pretrained_model_path, max_length=60,
              max_vocab_size=60000,
              embedding_dim=300,
              embedding_weight=None):
    
    # The embedding layer containing the word vectors
    emb_layer = Embedding(
        input_dim=max_vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_weight],
        input_length=max_length,
        trainable=True
    )
    
    # 1D convolutions that can iterate over the word vectors
    conv1 = Conv1D(filters=128, kernel_size=1, padding='same', activation='relu')
    conv2 = Conv1D(filters=128, kernel_size=2, padding='same', activation='relu')
    conv3 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')
    conv4 = Conv1D(filters=128, kernel_size=4, padding='same', activation='relu')
    conv5 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')
    conv6 = Conv1D(filters=32, kernel_size=6, padding='same', activation='relu')

    # Define inputs
    seq1 = Input(shape=(max_length,))
    seq2 = Input(shape=(max_length,))

    # Run inputs through embedding
    # emb1 = emb_layer(seq1)
    # emb2 = emb_layer(seq2)

    pretrained_model = PretrainedLSTM(pretrained_model_path,  return_sequences=False)

    emb1 = lstm(seq1)
    emb2 = lstm(seq2)

    # Run through CONV + GAP layers
    conv1a = conv1(emb1)
    glob1a = GlobalAveragePooling1D()(conv1a)
    conv1b = conv1(emb2)
    glob1b = GlobalAveragePooling1D()(conv1b)

    conv2a = conv2(emb1)
    glob2a = GlobalAveragePooling1D()(conv2a)
    conv2b = conv2(emb2)
    glob2b = GlobalAveragePooling1D()(conv2b)

    conv3a = conv3(emb1)
    glob3a = GlobalAveragePooling1D()(conv3a)
    conv3b = conv3(emb2)
    glob3b = GlobalAveragePooling1D()(conv3b)

    conv4a = conv4(emb1)
    glob4a = GlobalAveragePooling1D()(conv4a)
    conv4b = conv4(emb2)
    glob4b = GlobalAveragePooling1D()(conv4b)

    conv5a = conv5(emb1)
    glob5a = GlobalAveragePooling1D()(conv5a)
    conv5b = conv5(emb2)
    glob5b = GlobalAveragePooling1D()(conv5b)

    conv6a = conv6(emb1)
    glob6a = GlobalAveragePooling1D()(conv6a)
    conv6b = conv6(emb2)
    glob6b = GlobalAveragePooling1D()(conv6b)

    mergea = Concatenate()([glob1a, glob2a, glob3a, glob4a, glob5a, glob6a])
    mergeb = Concatenate()([glob1b, glob2b, glob3b, glob4b, glob5b, glob6b])

    # We take the explicit absolute difference between the two sentences
    # Furthermore we take the multiply different entries to get a different measure of equalness
    diff = Lambda(lambda x: K.abs(x[0] - x[1]), output_shape=(4 * 128 + 2*32,))([mergea, mergeb])
    mul = Lambda(lambda x: x[0] * x[1], output_shape=(4 * 128 + 2*32,))([mergea, mergeb])

    # # Add the magic features
    # magic_input = Input(shape=(5,))
    # magic_dense = BatchNormalization()(magic_input)
    # magic_dense = Dense(64, activation='relu')(magic_dense)

    # # Add the distance features (these are now TFIDF (character and word), Fuzzy matching, 
    # # nb char 1 and 2, word mover distance and skew/kurtosis of the sentence vector)
    # distance_input = Input(shape=(20,))
    # distance_dense = BatchNormalization()(distance_input)
    # distance_dense = Dense(128, activation='relu')(distance_dense)

    # # Merge the Magic and distance features with the difference layer
    # merge = concatenate([diff, mul, magic_dense, distance_dense])
    merge = Concatenate()([diff, mul])


    # The MLP that determines the outcome
    x = Dropout(0.2)(merge)
    x = BatchNormalization()(x)
    x = Dense(300, activation='relu')(x)

    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    pred = Dense(1, activation='sigmoid')(x)

    # model = Model(inputs=[seq1, seq2, magic_input, distance_input], outputs=pred)
    model = Model(inputs=[seq1, seq2], outputs=pred)
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    return model




def train_classifier(data_file, pretrained_model_path,  tokenizer="nltk"):
    maxlen = 60
    x_train, y_train , x_val , y_val, x_test, y_test = _load_quora_data(data_file, tokenizer=tokenizer, maxlen=maxlen)
    x_val = x_test
    y_val = y_test
    # Define Model
    model = get_model_v4(pretrained_model_path)

    # # for layer in pretrained_model.layers[:-1]:
    # #     layer.trainable = False
    # encoder_output = pretrained_model.outputs[0]
    # # encoder_output = Combine()(encoder_output)
    # # encoder_output = Lambda(combine_layer)(encoder_output)
    # # encoder_output = Dropout(0.5)(encoder_output)
    # # encoder_output = CuDNNLSTM(100)(encoder_output)
    # encoder_output = Dense(300, activation='relu')(encoder_output)
    # encoder_output = Dropout(0.5)(encoder_output)
    # encoder_output = Dense(50, activation='relu')(encoder_output)
    # encoder_output = Dropout(0.5)(encoder_output)
    # final_output = Dense(1, activation="sigmoid")(encoder_output)

    # model = Model(inputs=[input_], outputs=[final_output])
    model.compile("adam", loss="binary_crossentropy" , metrics=['acc'])
    class TableInitializerCallback(Callback):
        """ Initialize Tables """
        def on_train_begin(self, logs=None):
            K.get_session().run(tf.tables_initializer())
    callbacks = [TableInitializerCallback()]
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, callbacks=callbacks)

if __name__ == '__main__':
    fire.Fire(train_classifier)
