# encoding:utf-8
# import
from __future__ import absolute_import,division, print_function, unicode_literals
import sys
import tensorflow  as tf
import tensorflow_datasets as tfds
import tensorflow.keras.layers as layers
import time
import numpy as np
import matplotlib as plt
sys.path.append('.')
print(tf.__version__)

# use tfds load test data
examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True,download=False)
train_examples, val_examples = examples['train'], examples['validation']
# what is the new api for word encode decode
# https://blog.csdn.net/weixin_43788143/article/details/107902543
# 从训练数据集创建自定义子词分词器（subwords tokenizer）
# https://github.com/tensorflow/tensorflow/issues/45217
#create tokenizer
tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus((en.numpy() for pt, en in train_examples),
                                                                         target_vocab_size=2**13)
tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus((pt.numpy() for en, pt in train_examples),
                                                                         target_vocab_size=2**13)
# test
# sample  to encoding
sample_string = 'Transformer is awesome'
tokenized_string = tokenizer_en.encode(sample_string)
print('tokenized_string is {}'.format(tokenized_string))
original_string = tokenizer_en.decode(tokenized_string)
print('The original string is {}'.format(original_string))
assert original_string == sample_string
# 如果单词不在词典中，则分词器（tokenizer）通过将单词分解为子词来对字符串进行编码。
for ts in tokenized_string:
    print('{} ----> {}'.format(ts, tokenizer_en.decode([ts])))


# hyper_parameter
BUFFER_SIZE = 20000
BATCH_SIZE = 64


#?
# 将开始和结束标记（token）添加到输入和目标。
def encode(lang1, lang2):
    lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(
        lang1.numpy()) + [tokenizer_pt.vocab_size + 1]

    lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
        lang2.numpy()) + [tokenizer_en.vocab_size + 1]

    return lang1, lang2

# Note：为了使本示例较小且相对较快，删除长度大于40个标记的样本。
MAX_LENGTH = 40
def filter_max_length(x, y, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(x)<= max_length,
                         tf.size(y)<= max_length)

'''
.map() 内部的操作以图模式（graph mode）运行
，.map() 接收一个不具有 numpy 属性的图张量（graph tensor）
。该分词器（tokenizer）需要将一个字符串或 Unicode 符号，编码成整数。
因此，您需要在 tf.py_function 内部运行编码过程，tf.py_function 接收一个 eager 张量，
该 eager 张量有一个包含字符串值的 numpy 属性。
'''
def tf_encode(pt, en):
    result_pt, result_en = tf.py_function(encode, [pt, en], [tf.int64,tf.int64])
    result_pt.set_shape([None])
    result_en.set_shape([None])
    return result_pt, result_en

# train_dataset
train_dataset = train_examples.map(tf_encode)
train_dataset = train_dataset.filter(filter_max_length)
print('pause')


# 将数据集缓存到内存中以加快读取速度。
train_dataset = train_dataset.cache()
# https://blog.csdn.net/qq_32691667/article/details/104369570
# https://ithelp.ithome.com.tw/articles/10241789
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)
train_dataset =train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

val_dataset = val_examples.map(tf_encode)
val_dataset = val_dataset.filter(filter_max_length).padded_batch(BATCH_SIZE)

pt_batch, en_batch = next(iter(val_dataset))
print(pt_batch, en_batch)

# position encoding
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2) / np.float32(d_model)))
    return pos * angle_rates

def position_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    # sin in position indices is even , 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:,0::2])
    # cos in position indices is odd , 2i+1
    angle_rads[:,1::2] = np.cos(angle_rads[:, 1::2])
    position_encoding = angle_rads[np.newaxis,...]
    return tf.cast(position_encoding, dtype=tf.float32)

# position encoding layer
pos_encoding = position_encoding(50, 512)
print(pos_encoding.shape)

# plt.pcolormesh(pos_encoding[0], cmap='RdBu')
# plt.xlabel('Depth')
# plt.xlim((0, 512))
# plt.ylabel('Position')
# plt.colorbar()
# plt.show()

'''
遮挡（Masking）

遮挡一批序列中所有的填充标记（pad tokens）。
这确保了模型不会将填充作为输入。
该 mask 表明填充值 0 出现的位置：在这些位置 mask 输出 1，否则输出 0。
'''

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # 添加额外的维度来将填充加到
    # 注意力对数（logits）。
    return seq[:,tf.newaxis, tf.newaxis, :] # (batch_size, 1, 1, seq_len)

# test padding mask
x = tf.constant([[7,6,0,0,1],[1,2,3,0,0],[0,0,0,4,5]])
print(create_padding_mask(x))
print(" ")

'''
前瞻遮挡（look-ahead mask）用于遮挡一个序列中的后续标记（future tokens）。
换句话说，该 mask 表明了不应该使用的条目。

这意味着要预测第三个词，将仅使用第一个和第二个词。
与此类似，预测第四个词，仅使用第一个，第二个和第三个词，依此类推。
'''
def create_look_ahead_mask(size):
    # https://blog.csdn.net/ACM_hades/article/details/88790013
    mask = 1 - tf.linalg.band_part(tf.ones((size,size)), -1, 0)
    return mask

# test look_ahead_mask
x = tf.random.uniform((1,3))
temp = create_look_ahead_mask(x.shape[1])
print(temp)

# 按比缩放的点积注意力（Scaled dot product attention）
# Attention qkv formula
# img/attention_formula.sv
# https://tensorflow.google.cn/images/tutorials/transformer/scaled_attention.png
def scaled_dot_product_attention(q,k,v,mask):
    """计算注意力权重。
     q, k, v 必须具有匹配的前置维度。
     k, v 必须有匹配的倒数第二个维度，例如：seq_len_k = seq_len_v。
     虽然 mask 根据其类型（填充或前瞻）有不同的形状，
     但是 mask 必须能进行广播转换以便求和。

     参数:
       q: 请求的形状 == (..., seq_len_q, depth)
       k: 主键的形状 == (..., seq_len_k, depth)
       v: 数值的形状 == (..., seq_len_v, depth_v)
       mask: Float 张量，其形状能转换成
             (..., seq_len_q, seq_len_k)。默认为None。

     返回值:
       输出，注意力权重
     """
    matmul_qk = tf.matmul(q, k, transpose_b=True)
