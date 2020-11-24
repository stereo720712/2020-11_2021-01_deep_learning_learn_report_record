# load dependencies
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import random
import sys
import io
import zipfile
sys.path.append("..")
import tensorflow as tf
# data source
# utils
def load_data_jay_lyrics():
    """加载周杰伦歌词数据集"""
    with zipfile.ZipFile('./data/jaychou_lyrics.txt.zip') as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[0:10000]
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    return corpus_chars, corpus_indices, char_to_idx, idx_to_char, vocab_size


# load data
(corpus_chars, corpus_indices, char_to_idx, idx_to_char, vocab_size) = load_data_jay_lyrics()

maxlen = 40
step = 4
sentences = []
next_chars = []
# for loop that  defines one epoch, so in this case the number of sequences is one epoch
# each sequence having maxlen as the sequence of the sequence

for i in range(0, len(corpus_chars)-maxlen,step):
    sentences.append(corpus_chars[i:i+maxlen])
    next_chars.append(corpus_chars[i+maxlen])

print('nb sequences:', len(sentences))

print('Vectorization...')
# x is input data tensor
# y is target matrix
x = np.zeros((len(sentences),maxlen,len(idx_to_char)),dtype=np.bool)
y = np.zeros((len(sentences),len(idx_to_char)),dtype=np.bool)
for i , sentence in enumerate(sentences):
    for t , char in enumerate(sentence):
        x[i,t,char_to_idx[char]] = 1
    y[i, char_to_idx[next_chars[i]]] = 1

print('Build model...')
#define our model with LSTM layers and dropout
# model = Sequential()
# model.add(LSTM(128, input_shape=(maxlen, len(idx_to_char))))
# model.add(Dropout(.2))
# model.add(Dense(len(idx_to_char)))
# model.add(Activation('softmax'))
#
# # use RMSprop for gradient descent, and learning rate of .01
# optimizer = RMSprop(lr=0.01)
# # use cross entropy as loss function
# model.compile(loss='categorical_crossentropy', optimizer=optimizer)
# # train the LSTM model
# model.fit(x, y,
#           batch_size=128,
#           epochs=300)
#
# #save model
# model.save('lstm_country.h5')
# del model

# load model
from tensorflow.keras.models import load_model
model = load_model('lstm_country.h5')

# generate text
chavi = np.zeros((1, maxlen, vocab_size))
nextword = 0
cutOffSize = 4
# set random point in text as input
indexStart = np.random.randint(0, len(corpus_chars) - 41)
for i in range(maxlen):
    word = corpus_chars[indexStart + i]
    index = char_to_idx[word]
    chavi[0][i][index] = 1

# generate 200 words
for j in range(200):
    yhat = model.predict(chavi)
    print(idx_to_char[yhat.argmax()], end=' ')
    nextword = np.zeros((1, cutOffSize, vocab_size))
    flat = yhat.flatten()
    flat = flat.argsort()
    for k in range(cutOffSize):
        nextword[0][0][flat[-(1 + k)]] = 1

    for k in range(cutOffSize):
        chavi = np.delete(chavi, 0, 1)

    nextword = np.vstack((chavi[0], nextword[0]))

    chavi = np.array([nextword])
    #print(nextword)
