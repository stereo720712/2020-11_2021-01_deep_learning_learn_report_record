# LSTM Keras

import tensorflow as tf
from tensorflow import keras
import time
import math
import numpy as np
import sys
sys.path.append('.')

# LSTM class inherit Keras

class RNNModel(keras.layers.Layer):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.dense = keras.layers.Dense(vocab_size)

    def call(self, inputs, state):
        # inputs transpose to (num_steps,batch_size) then encoding
        X = tf.one_hot(tf.transpose(inputs),self.vocab_size)
        Y, state = self.rnn(X, state)
        # 全连接层会首先将Y的形状变成(num_steps * batch_size, num_hiddens)，它的输出
        # 形状为(num_steps * batch_size, vocab_size)
        output = self.dense(tf.reshape(Y, (-1, Y.shape[-1])))
        return output, state

    def get_init_state(self, *args, **kwargs):
        return self.rnn.cell.get_init_state(*args, **kwargs)
