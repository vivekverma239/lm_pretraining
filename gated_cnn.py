import numpy as np
import tensorflow as tf
from collections import namedtuple
from tensorflow.keras import layers

class GatedCNNBlock(object):

    def __init__(self, kernel_size, num_filters):
        self.conv = layers.Conv1D(num_filters, kernel_size, activation='relu') 
        self.conv_gate = layers.Conv1D(output_dim, kernel_size, activation="sigmoid")
        self.pad_input = layers.ZeroPadding1D(padding=(kernel_size-1, 0))

    def __call__(self, inputs):
        X = self.pad_input(inputs)
        A = self.conv(X)
        B = self.conv_gate(X)
        return layers.Multiply()([A, B])



class GatedCNN(object):

    def __init__(self, number_blocks=5, kernel_size=4, number_filters=800):
        self.blocks = []
        for _ in range(number_blocks):
            self.blocks.append(GatedCNNBlock(kernel_size, number_filters))

    
    def __call__(self, input):
        res_input, input = input, input 

        for block in self.blocks: 
            input = block(input)
            input += res_input
            res_input = input 
        
        return res_input
