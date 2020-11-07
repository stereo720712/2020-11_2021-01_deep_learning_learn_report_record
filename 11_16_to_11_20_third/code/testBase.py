import tensorflow as tf
import numpy as np
tf.__version__

X = tf.random.normal(shape=(3, 1))
W_xh = tf.random.normal(shape=(1, 4))
H = tf.random.normal(shape=(3, 4))
W_hh = tf.random.normal(shape=(4, 4))

X_M = tf.matmul(X,W_xh)
X_M