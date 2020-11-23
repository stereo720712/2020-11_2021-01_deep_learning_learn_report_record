import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as f
import numpy as np
import sys
import time
import zipfile
sys.path.append("..")
def load_data_jay_lyrics():
    '''load  jay lyrics'''
    with zipfile.ZipFile('./data/jaychou_lyrics.txt.zip') as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')
        corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
        corpus_chars = corpus_chars[0:10000]
        idx_to_char = list(set(corpus_chars))
        char_to_idx = dict([(char, i) for i , char in enumerate(idx_to_char)])
        vocab_size = len(char_to_idx)
        corpus_indices = [char_to_idx[char] for char in corpus_chars]
        return corpus_indices, char_to_idx, idx_to_char, vocab_size




# 每次采样的小批量的形状是(批量大小, 时间步数)
def to_onehot(X, size):
    # X shape: (batch), output shape:(batch, n_class)
    return [tf.one_hot(x, size, dtype=tf.float32) for x in X.T]


# initialize model paramter
def get_params(num_inputs, num_hiddens, num_outputs):
    def _one(shape):
        return tf.Variable(tf.random.normal(shape=shape, stddev=0.01, mean=0, dtype=tf.float32))

    # hidde layers parameter
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens,num_hiddens))
    b_h = tf.Variable(tf.zeros(num_hiddens), dtype=tf.float32)

    # output layers
    W_hq = _one(num_hiddens, num_outputs)
    b_q = tf.Variable(tf.zeros(num_outputs),dtype=tf.float32)
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    return  params



# main

(corpus_indices, char_to_idx, idx_to_char, vocab_size) = load_data_jay_lyrics()

## model hyperparameter
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size