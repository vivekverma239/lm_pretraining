import itertools
import os
import random
from text_processing import Tokenizer
import numpy as np
from nltk import word_tokenize
import spacy

nlp = spacy.load('en', disable=['parser', 'tagger', 'entity'])

def get_tokenizer(tokenizer):
    if tokenizer == "spacy":
        return lambda x: [str(i) for i in nlp(x)]
    else:
        return word_tokenize


def iterate(data, seq_length, var=20):
    while True:
        idx = 0
        while idx < data[0].shape[-1]:
            span = seq_length + random.randint(-var, var)
            temp =  data[0][:, idx:idx+span],data[1][:, idx:idx+span]
            idx += span
            yield temp

def load_and_process_data(train_file, valid_file, test_file=None,\
                          max_vocab_size=50000,
                          custom_tokenizer_function=None):
    """
        Load and Process data for Language Modelling Task

        Description: This module expects data in a text file, first sentences are
                        split on the basis on the basis of nextline ("\n")
                    - Next Step is Tokenization using keras Tokenizer Class
                    - Then Converting tokens to indexes

        :params:
                - train_file : Train file containing text sequences seperated by
                                nextline, this will be used for training
                - valid_file : Same as above used for validation
                - test_file : Same as above used for test file
                - max_vocab_size : Maximum Vocab size to consider
    """
    # Read and split the data
    train = open(train_file).read().split('\n')
    valid = open(valid_file).read().split('\n')

    # Tokenization
    filters = '' 
    tokenizer = Tokenizer(num_words=max_vocab_size,filters=filters,\
                            custom_tokenizer_function=custom_tokenizer_function)
    tokenizer.fit_on_texts(list(train) + list(valid))

    # Calculate word frequency, sorting it so word_freq[i] is for word which has "i" inded
    word_freq = [(0,1000)]+ [(v,tokenizer.word_counts[k]) for k,v in tokenizer.word_index.items()]
    word_freq = [i[1] for i in sorted(word_freq, key=lambda x : x[0])][:max_vocab_size]

    # Tokenize and convert into tokenid sequences
    word_index = tokenizer.word_index # Note this contains all the words (not trimmed)
    train_data = tokenizer.texts_to_sequences(train)
    valid_data = tokenizer.texts_to_sequences(valid)
    if test_file:
        # Processing for test data
        test = open(test_file).read().split('\n')
        test_data = tokenizer.texts_to_sequences(train)
        return word_freq, word_index, train_data, valid_data, test_data
    return word_freq, word_index, train_data, valid_data

def batchify(data, batch_size):
    flattened_data = list(itertools.chain(*data))
    max_len= int((len(flattened_data)-1)/(batch_size))*(batch_size)
    y = np.reshape(flattened_data[1:max_len+1], (batch_size,-1))
    X = np.reshape(flattened_data[:max_len], (batch_size,-1))
    return X, y
