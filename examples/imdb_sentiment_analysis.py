import os
import fire
from tqdm import tqdm
import random
import tensorflow as tf

from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
from models import PretrainedLSTM

# python -m data.imdb --save_path data/imdb --imdb_path data/imdb/aclImdb/
TEST_POS_DIR = 'test/pos'
TEST_NEG_DIR = 'test/neg'
TRAIN_POS_DIR = 'train/pos/'
TRAIN_NEG_DIR = 'train/neg/'



def process_data_for_classification(imdb_path, val_sample=5000 , tokenizer=None, maxlen=500):
    """
        Read the IMDB folder files and create language model training and
        validation file

        :params:
            - save_path: Path where to save lm_train.txt and lm_valid.txt
            - imdb_path: Root directory of imdb dataset
            - val_sample: Number of files to select as validation set
    """
    pos = [open(os.path.join(imdb_path, TRAIN_POS_DIR, i)).read() for i in\
                            tqdm(os.listdir(os.path.join(imdb_path, TRAIN_POS_DIR)))]
    neg = [open(os.path.join(imdb_path, TRAIN_NEG_DIR, i)).read() for i in \
                            tqdm(os.listdir(os.path.join(imdb_path, TRAIN_NEG_DIR)))]
    test_pos = [open(os.path.join(imdb_path, TEST_POS_DIR, i)).read() for i in \
                            tqdm(os.listdir(os.path.join(imdb_path, TEST_POS_DIR)))]
    test_neg = [open(os.path.join(imdb_path, TEST_NEG_DIR, i)).read() for i in \
                            tqdm(os.listdir(os.path.join(imdb_path, TEST_NEG_DIR)))]


    train = pos + neg
    test = test_pos + test_neg

    print('Test Data Creation..')
    val_index = random.sample(list(range(len(train))),val_sample)
    x_train = [i for idx, i in enumerate(train) if idx not in val_index]
    y_train = [1 if idx <= len(pos) else 0 for idx, i in enumerate(train) if idx not in val_index]
    x_val = [i for idx, i in enumerate(train) if idx in val_index]
    y_val = [1 if idx <= len(pos) else 0 for idx, i in enumerate(train) if idx in val_index]

    x_test = test
    y_test = [1]*len(test_pos) + [0]*len(test_neg)

    if not tokenizer:
        tokenizer = text_to_word_sequence
    x_train = list(map(tokenizer, x_train))
    x_val = list(map(tokenizer, x_val))
    x_test = list(map(tokenizer, x_test))

    custom_pad_sequences = lambda x: pad_sequences(x, maxlen=maxlen, dtype=object, padding='pre', value="-pad-")
    x_train = custom_pad_sequences(x_train)
    x_val = custom_pad_sequences( x_val)
    x_test = custom_pad_sequences(x_test)

    return x_train, y_train , x_val , y_val, x_train, y_train


def train_classifier(imdb_path, pretrained_model_path,  tokenizer=None):

    x_train, y_train , x_val , y_val, x_train, y_train = process_data_for_classification(imdb_path, tokenizer=tokenizer)

    # Define Model
    input_ = Input(shape=(500,), dtype=tf.string)
    pretrained_model = PretrainedLSTM(pretrained_model_path, input_)
    encoder_output = pretrained_model.outputs[0]
    final_output = Dense(1)(encoder_output)

    model = Model(inputs=[input_], outputs=[final_output])
    model.compile("adam", loss="binary_crossentropy" , metrics=['acc'])
    class TableInitializerCallback(Callback):
        """ Initialize Tables """
        def on_train_begin(self, logs=None):
            K.get_session().run(tf.tables_initializer())
    callbacks = [TableInitializerCallback()]
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, callbacks=callbacks)

if __name__ == '__main__':
    fire.Fire(train_classifier)
