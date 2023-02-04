# -*- coding: utf-8 -*-
"""

Convolutional Neural Networks for Sentence Classification
Date:    2018/9/28 15:14
"""

from keras.layers import Input
from keras.layers import Dense
from keras.layers import  Embedding
from keras.layers import GRU
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Conv1D
from keras.models import Model
from keras import optimizers
from keras import regularizers
from keras import backend as K
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers import core
from keras.layers import Reshape
from keras.layers import concatenate


def cnn_for_classification_multi_label(max_seq_len, input_dim, output_dim, embedding_matrix,
                                     num_class):
    """cnn_for_classification_multi_label"""
    inp = Input(shape=(max_seq_len,))
    x = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False)(inp)

    convs = []
    filter_sizes = [1, 2]

    for filter_size in filter_sizes:
        conv = Conv1D(filters=128, kernel_size=filter_size, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
        max_pool = GlobalMaxPooling1D()(conv)
        convs.append(max_pool)
        # avg_pool = GlobalAveragePooling1D()(conv)
        # convs.append(avg_pool)

    l_merge = concatenate(convs)

    outputs = []
    for i in range(num_class):
        x_i = Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.001))(l_merge)
        x_i = Dense(32, activation="relu", kernel_regularizer=regularizers.l2(0.001))(x_i)
        x_i = Dense(16, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x_i)
        output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01))(x_i)
        outputs.append(output)

    model = Model(inputs=inp, outputs=outputs)
    adam = optimizers.adam(clipvalue=1)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    return model