# -*- coding: utf-8 -*-
"""

Densely Connected CNN with Multi-scale Feature Attention for Text Classification
Date:    2018/9/28 15:14
"""

from keras.layers import Input
from keras.layers import Dense
from keras.layers import  Embedding
from keras.layers import GRU
from keras.layers import LSTM
from keras.layers import Add
from keras.layers import Conv1D
from keras.models import Model
from keras import optimizers
from keras import regularizers
from keras import backend as K
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers import core
from keras.layers import Reshape
from keras.layers import concatenate

from nlp_tasks.absa.models import keras_layers


def densely_cnn_multi_label(max_seq_len, input_dim, output_dim, embedding_matrix,
                                     num_class):
    """densely_cnn_multi_label"""
    inp = Input(shape=(max_seq_len,))
    x = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False)(inp)

    k = 128
    conv1 = Conv1D(filters=k, kernel_size=1, activation='relu', padding='same')(x)

    conv2 = Conv1D(filters=k, kernel_size=2, activation='relu', padding='same')(conv1)

    # conv3_1 = Conv1D(filters=k, kernel_size=2, activation='relu', padding='same')(conv1)
    # conv3_2 = Conv1D(filters=k, kernel_size=2, activation='relu', padding='same')(conv2)
    # conv3 = Add()([conv3_1, conv3_2])
    #
    # conv4_1 = Conv1D(filters=k, kernel_size=2, activation='relu', padding='same')(conv1)
    # conv4_2 = Conv1D(filters=k, kernel_size=2, activation='relu', padding='same')(conv2)
    # conv4_3 = Conv1D(filters=k, kernel_size=2, activation='relu', padding='same')(conv3)
    # conv4 = Add()([conv4_1, conv4_2, conv4_3])

    attention = keras_layers.DenselyCnnAttLayer()([conv1, conv2])
    outputs = []
    for i in range(num_class):
        x_i = keras_layers.AttLayer(50, i)(attention)
        x_i = Dense(40, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x_i)
        x_i = Dense(20, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x_i)
        output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01))(x_i)
        outputs.append(output)

    model = Model(inputs=inp, outputs=outputs)
    adam = optimizers.adam(clipvalue=1)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    return model