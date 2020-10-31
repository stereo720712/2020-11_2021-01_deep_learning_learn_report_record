# ref
# https://aigeekprogrammer.com/binary-classification-using-logistic-regression-and-keras/

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from tensorflow import keras
from keras.datasets import mnist


# Press the green button in the gutter to run the script.
# error
if __name__ == '__main__':
  print(keras.__version__)
  # 0 ~ 9 digit number pics
  (X_train, y_train), (X_test, y_test) = mnist.load_data()
  # pic size is 28*28
  X_train_re = X_train.reshape(-1, 28*28)/255
  X_test_re = X_test.reshape(-1, 28*28)/255

  model = keras.Sequential()
  model.add(keras.layers.Dense(1,input_shape=(784,),activation='sigmoid'))
  model.add(keras.layers.Dense(1,input_shape=(784,),activation='sigmoid'))
  model.compile(optimer='sgd', loss=keras.losses.CategoricalCrossentropy,
                metrics=['categorical_accuracy'])

  model.fit(x=X_train_re, y=y_train, shuffle=True, epochs=5,
            batch_size=16)

  # test accurancy
  eval = model.evaluate(x=X_test_re, y=y_test)
  print(eval)

