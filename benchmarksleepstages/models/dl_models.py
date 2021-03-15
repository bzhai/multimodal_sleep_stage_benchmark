"""
Make Sense of Sleep: Build the deep learning models
Copyright (C) 2020 Newcastle University, Bing Zhai
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from keras.layers import *
from keras.models import *
#
# def build_lstm_tf_v2(input_dim, num_classes):
#     model = tf.keras.Sequential()
#     model.add(tf.keras.layers.Input(input_dim))
#     model.add(tf.keras.layers.LSTM(units=32))
#     if num_classes <=2:
#         model.add(tf.keras.layers.Dense(num_classes, activation='sigmoid'))
#         entropy = 'binary_crossentropy'
#     else:
#         model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
#         entropy = 'categorical_crossentropy'
#     opt = tf.keras.optimizers.RMSprop()
#     model.compile(optimizer=opt, loss=entropy, metrics=['accuracy'])
#     model.summary()
#     return model
#
#
# def build_cnn_tf_v2(input_dim, num_classes):
#     model = tf.keras.Sequential()
#     model.add(tf.keras.layers.Input(input_dim))
#     model.add(tf.keras.layers.Convolution1D(filters=64, kernel_size=2, input_shape=input_dim))
#     model.add(tf.keras.layers.Activation('relu'))
#     model.add(tf.keras.layers.Flatten())
#     # if num_classes <=2:
#     #     model.add(tf.keras.layers.Dense(num_classes, activation='sigmoid'))
#     #     entropy = 'binary_crossentropy'
#     # else:
#     model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
#     entropy = 'categorical_crossentropy'
#     opt = tf.keras.optimizers.RMSprop()
#     model.compile(optimizer=opt, loss=entropy, metrics=['accuracy'])
#     model.summary()
#     return model


def build_lstm_tf_v1(input_dim, num_classes):
    """
    Build and return an LSTM model
    Args:
        input_dim(tuple): input shape(time_step, channels)
        num_classes(int): number of sleep stages
    """
    model = Sequential()
    model.add(LSTM(32, input_shape=input_dim))
    # model.add(Dropout(0.2))
    if num_classes <= 2:
        model.add(Dense(num_classes, activation='sigmoid'))
        entropy = 'binary_crossentropy'
    else:
        model.add(Dense(num_classes, activation='softmax'))
        entropy = 'categorical_crossentropy'
    model.compile(optimizer='rmsprop', loss=entropy, metrics=['accuracy'])
    model.summary()
    return model


def build_cnn_tf_v1(input_dim, num_classes):
    """
    Build and return an CNN model
    Args:
        input_dim(tuple): input shape(time_step, channels)
        num_classes(int): number of sleep stages
    """
    model = Sequential()
    model.add(Convolution1D(nb_filter=64, filter_length=2, input_shape=input_dim))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model
