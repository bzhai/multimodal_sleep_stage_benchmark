import tensorflow as tf
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


def build_lstm_tf_v1(input_dim, classes):
    model = Sequential()
    model.add(LSTM(32, input_shape=input_dim))
    #model.add(Dropout(0.2))
    if classes <= 2:
        model.add(Dense(classes, activation='sigmoid'))
        entropy = 'binary_crossentropy'
    else:
        model.add(Dense(classes, activation='softmax'))
        entropy = 'categorical_crossentropy'
    model.compile(optimizer='rmsprop', loss=entropy, metrics=['accuracy'])

    model.summary()
    return model


def build_cnn_tf_v1(input_dim, num_classes):
    model = Sequential()
    model.add(Convolution1D(nb_filter=64, filter_length=2, input_shape=input_dim))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    return model
