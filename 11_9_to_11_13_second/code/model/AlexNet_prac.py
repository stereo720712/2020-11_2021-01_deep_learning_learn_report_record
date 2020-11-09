from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import Adam

def AlexNet(input_shape=(224, 224, 3), output_shape=2):
    #AlexNet
    net = Sequential()
    net.add(Conv2D(filters=48, kernel_size=11, strides=4, activation='relu', input_shape=input_shape))
    net.add(BatchNormalization())
    net.add(MaxPool2D(pool_size=3, strides=2))
    net.add(Conv2D(filters=128, kernel_size=5, padding='same', activation='relu'))
    net.add(BatchNormalization())
    net.add(MaxPool2D(pool_size=3, strides=2))
    net.add(Conv2D(filters=192, kernel_size=3, padding='same', activation='relu'))
    net.add(Conv2D(filters=192, kernel_size=3, padding='same', activation='relu'))
    net.add(Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
    net.add(MaxPool2D(pool_size=3, strides=2))
    net.add(Flatten())
    net.add(Dense(4096, activation='relu'))
    net.add(Dropout(0.5))
    net.add(Dense(4096, activation='relu'))
    net.add(Dropout(0.5))
    net.add(Dense(10, activation='sigmoid'))
