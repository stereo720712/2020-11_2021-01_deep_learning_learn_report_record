import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as f
import numpy as np
import sys
import time
import zipfile
import random
import math
sys.path.append(".")
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

def data_iter_random(corpus_indices, batch_size, num_steps, ctx=None):
    # 减1是因为输出的索引是相应输入的索引加1
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    # 返回从pos开始的长为num_steps的序列
    def _data(pos):
        return corpus_indices[pos: pos + num_steps]

    for i in range(epoch_size):
        # 每次读取batch_size个随机样本
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices]
        yield np.array(X, ctx), np.array(Y, ctx)

def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None):
    corpus_indices = np.array(corpus_indices)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0: batch_size*batch_len].reshape((
        batch_size, batch_len))
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y

# 每次采样的小批量的形状是(批量大小, 时间步数)
def to_onehot(X, size):
    # X shape: (batch), output shape:(batch, n_class)
    return [tf.one_hot(x, size, dtype=tf.float32) for x in X.T]


# 计算裁剪后的梯度
def grad_clipping(grads,theta):
    norm = np.array([0])
    for i in range(len(grads)):
        norm+=tf.math.reduce_sum(grads[i] ** 2)
    #print("norm",norm)
    norm = np.sqrt(norm).item()
    new_gradient=[]
    if norm > theta:
        for grad in grads:
            new_gradient.append(grad * theta / norm)
    else:
        for grad in grads:
            new_gradient.append(grad)
    #print("new_gradient",new_gradient)
    return new_gradient


# model pre define

## initialize model paramter
def get_params():
    def _one(shape):
        return tf.Variable(tf.random.normal(shape=shape, stddev=0.01, mean=0, dtype=tf.float32))

    # hidde layers parameter
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens,num_hiddens))
    b_h = tf.Variable(tf.zeros(num_hiddens), dtype=tf.float32)

    # output layers
    W_hq = _one((num_hiddens, num_outputs))
    b_q = tf.Variable(tf.zeros(num_outputs),dtype=tf.float32)
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    return  params

## initial rnn hidden sate
def init_rnn_state(batch_size, num_hiddens):
    return (tf.zeros(shape=(batch_size, num_hiddens)), )


## model core
def rnn(inputs, state, params):
    # inputs和outputs皆为num_steps个形状为(batch_size, vocab_size)的矩阵
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        X=tf.reshape(X,[-1,W_xh.shape[0]])
        H = tf.tanh(tf.matmul(X, W_xh) + tf.matmul(H, W_hh) + b_h)
        Y = tf.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)

## predict function
def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,
                num_hiddens, vocab_size,idx_to_char, char_to_idx):

    # initial
    state = init_rnn_state(1, num_hiddens)
    output = [char_to_idx[prefix[0]]]

    for t in range(num_chars + len(prefix) - 1):
        # 将上一时间步的输出作为当前时间步的输入
        X = tf.convert_to_tensor(to_onehot(np.array([output[-1]]), vocab_size),dtype=tf.float32)
        X = tf.reshape(X,[1,-1])
        # 计算输出和更新隐藏状态
        (Y, state) = rnn(X, state, params)
        # 下一个时间步的输入是prefix里的字符或者当前的最佳预测字符
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(np.array(tf.argmax(Y[0],axis=1))))
    #print(output)
    #print([idx_to_char[i] for i in output])
    return ''.join([idx_to_char[i] for i in output])

## train function
def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          vocab_size,  corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps,
                          lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes):
    if is_random_iter:
        data_iter_fn = data_iter_random
    else:
        data_iter_fn = data_iter_consecutive
    params = get_params()
    #loss = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

    for epoch in range(num_epochs):
        if not is_random_iter:  # 如使用相邻采样，在epoch开始时初始化隐藏状态
            state = init_rnn_state(batch_size, num_hiddens)
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps)
        for X, Y in data_iter:
            if is_random_iter:  # 如使用随机采样，在每个小批量更新前初始化隐藏状态
                state = init_rnn_state(batch_size, num_hiddens)
            #else:  # 否则需要使用detach函数从计算图分离隐藏状态
                #for s in state:
                    #s.detach()
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(params)
                inputs = to_onehot(X, vocab_size)
                # outputs有num_steps个形状为(batch_size, vocab_size)的矩阵
                (outputs, state) = rnn(inputs, state, params)
                # 拼接之后形状为(num_steps * batch_size, vocab_size)
                outputs = tf.concat(outputs, 0)
                # Y的形状是(batch_size, num_steps)，转置后再变成长度为
                # batch * num_steps 的向量，这样跟输出的行一一对应
                y = Y.T.reshape((-1,))
                #print(Y,y)
                y=tf.convert_to_tensor(y,dtype=tf.float32)
                # 使用交叉熵损失计算平均分类误差
                l = tf.reduce_mean(tf.losses.sparse_categorical_crossentropy(y,outputs))
                #l = loss(y,outputs)
                #print("loss",np.array(l))

            grads = tape.gradient(l, params)
            grads=grad_clipping(grads, clipping_theta)  # 裁剪梯度
            optimizer.apply_gradients(zip(grads, params))
            #sgd(params, lr, 1 , grads)  # 因为误差已经取过均值，梯度不用再做平均
            l_sum += np.array(l).item() * len(y)
            n += len(y)

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            #print(params)
            for prefix in prefixes:
                print(prefix)
                print(' -', predict_rnn(
                    prefix, pred_len, rnn, params, init_rnn_state,
                    num_hiddens, vocab_size,  idx_to_char, char_to_idx))

# main

(corpus_indices, char_to_idx, idx_to_char, vocab_size) = load_data_jay_lyrics()

## model hyperparameter
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
num_epochs, num_steps, batch_size, lr, clipping_theta = 250, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']

# 隨機採樣
train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                      vocab_size, corpus_indices, idx_to_char,
                      char_to_idx, True, num_epochs, num_steps, lr,
                      clipping_theta, batch_size, pred_period, pred_len,
                      prefixes)
#相鄰採樣
train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                      vocab_size, corpus_indices, idx_to_char,
                      char_to_idx, False, num_epochs, num_steps, lr,
                      clipping_theta, batch_size, pred_period, pred_len,
                      prefixes)