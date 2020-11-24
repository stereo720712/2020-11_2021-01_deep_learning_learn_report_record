import tensorflow as tf
from tensorflow import keras
import time
import math
import numpy as np
import sys
import zipfile
import random
sys.path.append("..")

# =========================utils
# load data
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

def to_onehot(X, size):
    # X shape: (batch), output shape: (batch, n_class)
    return [tf.one_hot(x, size,dtype=tf.float32) for x in X.T]


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


# predict rnn
def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,
                num_hiddens, vocab_size,idx_to_char, char_to_idx):
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


# train predict rnn
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

def predict_rnn_keras(prefix, num_chars, model, vocab_size, idx_to_char,
                      char_to_idx):
    # 使用model的成员函数来初始化隐藏状态
    state = model.get_initial_state(batch_size=1,dtype=tf.float32)
    output = [char_to_idx[prefix[0]]]
    #print("output:",output)
    for t in range(num_chars + len(prefix) - 1):
        X = np.array([output[-1]]).reshape((1, 1))
        #print("X",X)
        Y, state = model(X, state)  # 前向计算不需要传入模型参数
        #print("Y",Y)
        #print("state:",state)
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
            #print(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(np.array(tf.argmax(Y,axis=-1))))
            #print(int(np.array(tf.argmax(Y[0],axis=-1))))
    return ''.join([idx_to_char[i] for i in output])

def train_and_predict_rnn_keras(model, num_hiddens, vocab_size,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes):
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer=tf.keras.optimizers.SGD(learning_rate=lr)

    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_consecutive(
            corpus_indices, batch_size, num_steps)
        state = model.get_initial_state(batch_size=batch_size,dtype=tf.float32)
        for X, Y in data_iter:
            with tf.GradientTape(persistent=True) as tape:
                (outputs, state) = model(X, state)
                y = Y.T.reshape((-1,))
                l = loss(y,outputs)

            grads = tape.gradient(l, model.variables)
            # 梯度裁剪
            grads=grad_clipping(grads, clipping_theta)
            optimizer.apply_gradients(zip(grads, model.variables))  # 因为已经误差取过均值，梯度不用再做平均
            l_sum += np.array(l).item() * len(y)
            n += len(y)

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn_keras(
                    prefix, pred_len, model, vocab_size,  idx_to_char,
                    char_to_idx))

# =========================utils

(corpus_indices, char_to_idx, idx_to_char,vocab_size) = load_data_jay_lyrics()
# initial model parameter
## hyperparameter num_hiddens ,隱藏單元個數
num_inputs, num_hiddens, num_outputs = vocab_size, 128, vocab_size
def get_params():
    def _one(shape):
        return tf.Variable(tf.random.normal(shape=shape,stddev=0.01,mean=0,dtype=tf.float32))
    def _three():
        return (_one((num_inputs, num_hiddens)),
                _one((num_hiddens, num_hiddens)),
                tf.Variable(tf.zeros(num_hiddens), dtype=tf.float32))
    W_xi, W_hi, b_i = _three()  # 输入门参数
    W_xf, W_hf, b_f = _three()  # 遗忘门参数
    W_xo, W_ho, b_o = _three()  # 输出门参数
    W_xc, W_hc, b_c = _three()  # 候选记忆细胞参数

    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = tf.Variable(tf.zeros(num_outputs), dtype=tf.float32)
    return [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q]




# define model

## memory cell initial
def init_lstm_state(batch_size, num_hiddens):
    return (tf.zeros(shape=(batch_size, num_hiddens)),
            tf.zeros(shape=(batch_size, num_hiddens)))

## lstm core
def lstm(inputs, state, params):
    W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q = params
    (H, C) = state
    outputs = []
    for X in inputs:
        X=tf.reshape(X,[-1,W_xi.shape[0]])
        I = tf.sigmoid(tf.matmul(X, W_xi) + tf.matmul(H, W_hi) + b_i)
        F = tf.sigmoid(tf.matmul(X, W_xf) + tf.matmul(H, W_hf) + b_f)
        O = tf.sigmoid(tf.matmul(X, W_xo) + tf.matmul(H, W_ho) + b_o)
        C_tilda = tf.tanh(tf.matmul(X, W_xc) + tf.matmul(H, W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * tf.tanh(C)
        Y = tf.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H, C)

# train and predict
num_epochs, num_steps, batch_size, lr, clipping_theta = 160, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 40, 50, ['分开', '不分开']

train_and_predict_rnn(lstm, get_params, init_lstm_state, num_hiddens,
                          vocab_size, corpus_indices, idx_to_char,
                          char_to_idx, False, num_epochs, num_steps, lr,
                          clipping_theta, batch_size, pred_period, pred_len,
                          prefixes)


#== keras
class RNNModel(keras.layers.Layer):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.dense = keras.layers.Dense(vocab_size)

    def call(self, inputs, state):
        # 将输入转置成(num_steps, batch_size)后获取one-hot向量表示
        X = tf.one_hot(tf.transpose(inputs), self.vocab_size)
        Y,state = self.rnn(X, state)
        # 全连接层会首先将Y的形状变成(num_steps * batch_size, num_hiddens)，它的输出
        # 形状为(num_steps * batch_size, vocab_size)
        output = self.dense(tf.reshape(Y,(-1, Y.shape[-1])))
        return output, state

    def get_initial_state(self, *args, **kwargs):
        return self.rnn.cell.get_initial_state(*args, **kwargs)

lr = 1e-2 # 注意调整学习率
lstm_layer = keras.layers.LSTM(num_hiddens,time_major=True,return_sequences=True,return_state=True)
model = RNNModel(lstm_layer, vocab_size)
train_and_predict_rnn_keras(model, num_hiddens, vocab_size,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes)