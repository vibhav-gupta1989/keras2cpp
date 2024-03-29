'''
Save CNN network and one sample of train data.

Run one iteration of training of convnet on the MNIST dataset.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Conv2D
from keras.utils import to_categorical

batch_size = 128
nb_classes = 10
nb_epoch = 1

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 4
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train_original = X_train
X_train_original = X_train_original.astype('float32')
X_train_original /= 255

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = to_categorical(y_train, nb_classes)
Y_test = to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Conv2D(nb_filters, nb_conv, padding='same',
                        input_shape=(img_rows, img_cols, 1)))
#model.add(Conv2D(nb_filters, nb_conv, padding = 'same',
#                 input_shape = (1, img_rows, img_cols )))
model.add(Activation('relu'))
model.add(Conv2D(nb_filters, nb_conv, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(6))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta')

model.summary()

model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))

# store model
# with open('./my_nn_arch.json', 'w') as fout:
#     fout.write(model.to_json())
# model.save_weights('./my_nn_weights.h5', overwrite=True, save_format='h5')
model.save("my_model_mnist.keras")

# store one sample in text file
with open("./sample_mnist.dat", "w") as fin:
    fin.write("28 28 1\n")
    a = X_train_original[4]
    for b in a:
        fin.write(str(b)+'\n')

# get prediction on saved sample
# c++ output should be the same ;)
print('Prediction on saved sample:')
#print(X_train[1:2])
print(str(model.predict(X_train[4:5])))
# on my pc I got:
#[[ 0.03729606  0.00783805  0.06588034  0.21728528  0.01093729  0.34730983
#   0.01350389  0.02174525  0.26624694  0.01195715]]

