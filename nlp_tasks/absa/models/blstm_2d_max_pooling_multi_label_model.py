# -*- coding: utf-8 -*-
"""

Text Classification Improved by Integrating Bidirectional LSTM with Two-dimensional Max Pooling
Date:    2018/9/28 15:14
"""

from keras.layers import Input
from keras.layers import Dense
from keras.layers import  Embedding
from keras.layers import GRU
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Conv2D
from keras.models import Model
from keras import optimizers
from keras import regularizers
from keras import backend as K
from keras.layers.pooling import GlobalMaxPooling2D
from keras.layers import core
from keras.layers import Reshape


def blstm_2d_max_pooling_multi_label(max_seq_len, input_dim, output_dim, embedding_matrix,
                                     num_class):
    """blstm_2d_max_pooling_multi_label"""
    inp = Input(shape=(max_seq_len,))
    x = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False)(inp)
    # x = SpatialDropout1D(0.2)(x)
    # x = Bidirectional(GRU(40, return_sequences=True, input_length=max_seq_len,
    #                       kernel_regularizer=regularizers.l2(0.01)))(x)
    # x = Bidirectional(GRU(20, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))(x)
    # x = Bidirectional(LSTM(40, return_sequences=True, dropout=0.5))(x)

    x = Reshape((x.get_shape().as_list()[1], x.get_shape().as_list()[2], 1))(x)
    conv = Conv2D(filters=128, kernel_size=(2, 2), activation='relu')(x)
    pool = GlobalMaxPooling2D()(conv)
    outputs = []
    for i in range(num_class):
        x_i = Dense(20, activation="relu", kernel_regularizer=regularizers.l2(0.01))(pool)
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