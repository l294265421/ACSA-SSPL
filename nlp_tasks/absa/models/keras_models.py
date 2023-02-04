from keras.layers import Input, Dense, Embedding, SpatialDropout1D, Add, Concatenate
from keras.layers import GRU, LSTM, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, \
    concatenate, CuDNNGRU
from keras.models import Model
from keras import optimizers
from keras import regularizers
from keras import losses
from keras.layers import Flatten
from keras.layers import Dot
from keras.layers import Multiply
from keras.layers.wrappers import TimeDistributed
from keras_bert import load_trained_model_from_checkpoint
import tensorflow as tf

import os
os.environ["PATH"] += os.pathsep + 'D:/Program Files (x86)/Graphviz2.38/bin/'

# import pydot_ng as pydot
# print (pydot.find_graphviz())

from nlp_tasks.absa.conf import data_path
from nlp_tasks.absa.conf import task_conf


def log_tensor(t,  description=''):
    return Lambda(lambda x: tf.Print(x, [x], description, summarize=400000))(t)


def aae_aoa_block(vecs, layer_index=-1, aspect_index=-1):
    attend_weight = Attention(use_W=True, use_bias=False, u_size=5, W_regularizer=regularizers.l2(0.01),
                              u_regularizer=regularizers.l2(0.01),
                              name=('ac_attention_%d_%d' % (layer_index, aspect_index)))(vecs)
    aspect_att = generate_attention_output_by_attention_weight(vecs, attend_weight)

    attend_weight = Lambda(ac_aoa, name=('se_attention_%d_%d' % (layer_index, aspect_index)))([aspect_att, vecs])
    sentiment_att = generate_attention_output_by_attention_weight(vecs, attend_weight)
    return aspect_att, sentiment_att


def aae_complete_attention_block(vecs, layer_index=-1, aspect_index=-1, W_dense=None):
    attend_weight = Attention(use_W=True, use_bias=False, u_size=5, W_regularizer=regularizers.l2(0.01),
                              u_regularizer=regularizers.l2(0.01))(vecs)
    if task_conf.log_attention:
        attend_weight = log_tensor(attend_weight, 'ac_%d_%d' % (layer_index, aspect_index))
    aspect_att = generate_attention_output_by_attention_weight(vecs, attend_weight)

    # 通过aspect_att作为query去vecs查相关信息
    attend_weight = Lambda(ac_aoa_wrapper(W_dense=W_dense))([aspect_att, vecs])

    if task_conf.log_attention:
        attend_weight = log_tensor(attend_weight, 'se_%d_%d' % (layer_index, aspect_index))
    sentiment_att = generate_attention_output_by_attention_weight(vecs, attend_weight)
    return aspect_att, sentiment_att


def aae_at_lstm_block(vecs, max_len, share_attenion_layer):
    attend_weight = Attention(use_W=True, use_bias=False, u_size=5, W_regularizer=regularizers.l2(0.01),
                              u_regularizer=regularizers.l2(0.01))(vecs)
    aspect_att = generate_attention_output_by_attention_weight(vecs, attend_weight)

    aspect_embed = RepeatVector(max_len)(aspect_att)
    concat = concatenate([vecs, aspect_embed], axis=-1)
    attend_weight = share_attenion_layer(concat)
    sentiment_att = generate_attention_output_by_attention_weight(vecs, attend_weight)
    return aspect_att, sentiment_att


def aoa_block_with_fixed_ae(vecs, ae):
    attend_weight = Attention(use_W=True, use_bias=False, u_size=5, W_regularizer=regularizers.l2(0.01),
                              u_regularizer=regularizers.l2(0.01))(vecs)
    aspect_att = generate_attention_output_by_attention_weight(vecs, attend_weight)

    attend_weight = Lambda(ac_aoa)([ae, vecs])
    sentiment_att = generate_attention_output_by_attention_weight(vecs, attend_weight)
    return aspect_att, sentiment_att


def aoa_block_no_com(vecs):
    attend_weight = Attention(use_W=True, use_bias=False, u_size=5, W_regularizer=regularizers.l2(0.01),
                              u_regularizer=regularizers.l2(0.01))(vecs)
    aspect_att = generate_attention_output_by_attention_weight(vecs, attend_weight)

    aae = Lambda(lambda x: K.stop_gradient(x))(aspect_att)

    attend_weight = Lambda(ac_aoa)([aae, vecs])
    sentiment_att = generate_attention_output_by_attention_weight(vecs, attend_weight)
    return aspect_att, sentiment_att


def rcnn(max_seq_len, input_dim, output_dim, embedding_matrix):
    """rcnn"""
    inp = Input(shape=(max_seq_len,))
    x = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(0.5)(x)
    x = Bidirectional(LSTM(40, return_sequences=True))(x)
    x = Bidirectional(GRU(40, return_sequences=True))(x)
    # avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    # conc = concatenate([avg_pool, max_pool])
    outp = Dense(30, activation="sigmoid")(max_pool)

    model = Model(inputs=inp, outputs=outp)
    adam = optimizers.adam(clipvalue=0.5)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    return model


def rnn_attention(max_seq_len, input_dim, output_dim, embedding_matrix, num_class=10):
    """rnn_attention"""
    inp = Input(shape=(max_seq_len,))
    x = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(0.2)(x)
    # x = Bidirectional(LSTM(60, return_sequences=True))(x)
    x = Bidirectional(GRU(60, return_sequences=True))(x)
    x = keras_layers.AttLayer(50, 0)(x)
    # avg_pool = GlobalAveragePooling1D()(x)
    # max_pool = GlobalMaxPooling1D()(x)
    # conc = concatenate([avg_pool, max_pool])
    outp = Dense(num_class, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=outp)
    adam = optimizers.adam(clipvalue=1)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    return model


def at_lstm(max_seq_len, input_dim, output_dim, embedding_matrix, num_class=10):
    """at_lstm"""
    word_input = Input(shape=(max_seq_len,), name='word_input')
    x_word = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding')(word_input)

    x_word = SpatialDropout1D(0.2, name='SpatialDropout1D')(x_word)
    x_word, last_state_0, last_state_1 = Bidirectional(GRU(60, return_sequences=True,
                               return_state=True,
                               name='gru',
                               kernel_regularizer=regularizers.l2(0.01),
                               bias_regularizer=regularizers.l2(0.01)))(x_word)

    aspect_input = Input(shape=(max_seq_len,), name='aspect_input')
    x_aspect = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                         name='aspect_embedding')(aspect_input)

    x_word_and_aspect = concatenate([x_word, x_aspect])

    x = keras_layers.AttLayerDifferentKeyValue(50, name='AttLayerDifferentKeyValue')(
        [x_word_and_aspect, x_word])
    # x = concatenate([x, last_state_0, last_state_1])
    outp = Dense(num_class, activation="softmax", name='dense', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(x)

    model = Model(inputs=[word_input, aspect_input], outputs=outp, name='model')
    adam = optimizers.adam(clipvalue=1)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    return model


def joint_model_of_aspect_and_sentiment(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []

    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    x_word = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding')(word_input)
    x_word = SpatialDropout1D(0.2, name='SpatialDropout1D')(x_word)
    x_word, last_state_0, last_state_1 = Bidirectional(GRU(60, return_sequences=True,
                               return_state=True,
                               name='gru',
                               kernel_regularizer=regularizers.l2(0.01),
                               bias_regularizer=regularizers.l2(0.01)))(x_word)
    aspect_attentions = []
    for i in range(aspect_class_num):
        aspect_att_i = keras_layers.AttLayer(20, i)(x_word)
        aspect_attentions.append(aspect_att_i)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01))(aspect_att_i)
        outputs.append(aspect_output)
        loss.append('binary_crossentropy')

    sentiment_attention = keras_layers.AttLayerDifferentKeyValue(50, name='AttLayerDifferentKeyValue')
    sentiment_out = Dense(sentiment_class_num, activation="softmax", name='aspect_dense_' +  str(i),
                                 kernel_regularizer=regularizers.l2(0.01),
                                 bias_regularizer=regularizers.l2(0.01)
                                 )
    for i in range(aspect_class_num):
        aspect_input = Input(shape=(max_seq_len,), name='aspect_input_' + str(i))
        inputs.append(aspect_input)
        x_aspect = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                             name='aspect_embedding_' + str(i))(aspect_input)
        x_word_and_aspect = concatenate([x_word, x_aspect])

        sentiment_att_i = sentiment_attention([x_word_and_aspect, x_word])
        aspect_att_and_sentiment_att = Add()([aspect_attentions[i], sentiment_att_i])
        sentiment_output = sentiment_out(aspect_att_and_sentiment_att)
        outputs.append(sentiment_output)
        loss.append('categorical_crossentropy')

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=1)
    model.compile(loss=loss,
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    return model


from nlp_tasks.absa.models.custom_layers import Attention, TopicAttention, RecurrentAttention, InteractiveAttention, ContentAttention
from keras.layers import Input, Embedding, SpatialDropout1D, Dropout, Conv1D, MaxPool1D, Flatten, concatenate, Dense, \
    LSTM, Bidirectional, Activation, MaxPooling1D, Add, GRU, GlobalAveragePooling1D, GlobalMaxPooling1D, RepeatVector, \
    TimeDistributed, Permute, multiply, Lambda, add, Masking, BatchNormalization, Softmax, Reshape, ReLU, \
    ZeroPadding1D, subtract
import keras.backend as K
from nlp_tasks.absa.models import keras_layers, capsulelayers


def joint_model_of_aspect_and_sentiment_semeval2014(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding')(word_input)

    # aspect category分类
    # text_embed = SpatialDropout1D(0.2, name='SpatialDropout1D')(word_embedding)
    # hidden_vecs = GRU(50, return_sequences=True)(text_embed)
    hidden_vecs, last_state_0, last_state_1 = Bidirectional(GRU(128, return_sequences=True,
                                                           return_state=True,
                                                           name='gru',
                                                           # kernel_regularizer=regularizers.l2(0.01),
                                                           # bias_regularizer=regularizers.l2(0.01)
                                                                ))(word_embedding)
    attend_weight = Attention(use_W=False)(hidden_vecs)
    attend_weight_expand = Lambda(lambda x: K.expand_dims(x))(attend_weight)
    attend_hidden = multiply([hidden_vecs, attend_weight_expand])
    aspect_att = Lambda(lambda x: K.sum(x, axis=1))(attend_hidden)
    dense = Dense(64, activation='relu')(aspect_att)
    for i in range(aspect_class_num):
        final_output = Dense(1, activation="sigmoid")(dense)
        outputs.append(final_output)
        loss.append('binary_crossentropy')
        loss_weights.append(1)

    # 主题对应的情感分类
    hidden_vecs_sentiment, last_state_0_sentiment, last_state_1_sentiment = Bidirectional(GRU(25, return_sequences=True,
                                                                return_state=True,
                                                                name='hidden_vecs_sentiment',
                                                                # kernel_regularizer=regularizers.l2(0.01),
                                                                # bias_regularizer=regularizers.l2(0.01)
                                                                ))(word_embedding)
    hidden_vecs_sentiment = SpatialDropout1D(0.5, name='SpatialDropout1D_sentiment')(hidden_vecs_sentiment)
    sentiment_attention = Attention(W_regularizer=regularizers.l2(0.01),
                                    u_regularizer=regularizers.l2(0.01))
    asp_embedding = Embedding(input_dim=6, output_dim=50, trainable=True)
    # sentiment_dense = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))
    sentiment_out = Dense(sentiment_class_num, activation='softmax',
                          kernel_regularizer=regularizers.l2(0.1))
    for i in range(aspect_class_num):
        input_aspect = Input(shape=(1,), )
        inputs.append(input_aspect)
        aspect_embed = asp_embedding(input_aspect)
        # reshape to 2d
        aspect_embed = Flatten()(aspect_embed)
        # repeat aspect for every word in sequence
        repeat_aspect = RepeatVector(max_seq_len)(aspect_embed)

        # mask after concatenate will be same as hidden_out's mask
        concat = concatenate([hidden_vecs_sentiment, repeat_aspect], axis=-1)

        # apply attention mechanism
        attend_weight = sentiment_attention(concat)
        attend_weight_expand = Lambda(lambda x: K.expand_dims(x))(attend_weight)
        attend_hidden = multiply([hidden_vecs_sentiment, attend_weight_expand])
        attend_hidden = Lambda(lambda x: K.sum(x, axis=1))(attend_hidden)

        # aspect_att_and_sentiment_att = Add()([aspect_attentions[i], attend_hidden])
        # aspect_att_and_sentiment_att = attend_hidden
        # aspect_att_and_sentiment_att = Dropout(0.3, name='Dropout_' + str(i))(aspect_att_and_sentiment_att)
        # dense_layer = sentiment_dense(aspect_att_and_sentiment_att)
        # aspect_att_and_sentiment_att = Dropout(0.3, name='Dropout_' + str(i))(aspect_att_and_sentiment_att)
        output_layer = sentiment_out(attend_hidden)
        outputs.append(output_layer)
        loss.append('categorical_crossentropy')
        loss_weights.append(1)

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam()
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    return model


def joint_model_of_aspect_and_sentiment_semeval2014_taws(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding')(word_input)

    # aspect category分类
    # text_embed = SpatialDropout1D(0.2, name='SpatialDropout1D')(word_embedding)
    # hidden_vecs = GRU(50, return_sequences=True)(text_embed)
    hidden_vecs, last_state_0, last_state_1 = Bidirectional(GRU(128, return_sequences=True,
                                                           return_state=True,
                                                           name='gru',
                                                           # kernel_regularizer=regularizers.l2(0.01),
                                                           # bias_regularizer=regularizers.l2(0.01)
                                                                ))(word_embedding)
    # hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D')(hidden_vecs)
    # hidden_vecs = Dropout(0.3, name='Dropout')(hidden_vecs)
    aspect_attentions = []
    denses = []
    for i in range(aspect_class_num):
        # apply attention mechanism
        attend_weight = Attention(use_W=False)(hidden_vecs)
        attend_weight_expand = Lambda(lambda x: K.expand_dims(x))(attend_weight)
        attend_hidden = multiply([hidden_vecs, attend_weight_expand])
        aspect_att_i = Lambda(lambda x: K.sum(x, axis=1))(attend_hidden)

        aspect_attentions.append(aspect_att_i)
        dense = Dense(16, activation='relu')(aspect_att_i)
        denses.append(dense)

    denses_concat = concatenate(denses)
    for i in range(aspect_class_num):
        # denses_concat = Dropout(0.4)(denses_concat)
        final_output = Dense(1, activation="sigmoid")(
            denses_concat)
        outputs.append(final_output)
        loss.append('binary_crossentropy')
        loss_weights.append(1)

    # 主题对应的情感分类
    hidden_vecs_sentiment, last_state_0_sentiment, last_state_1_sentiment = Bidirectional(GRU(25, return_sequences=True,
                                                                return_state=True,
                                                                name='hidden_vecs_sentiment',
                                                                # kernel_regularizer=regularizers.l2(0.01),
                                                                # bias_regularizer=regularizers.l2(0.01)
                                                                ))(word_embedding)
    hidden_vecs_sentiment = SpatialDropout1D(0.5, name='SpatialDropout1D_sentiment')(hidden_vecs_sentiment)
    sentiment_attention = Attention(W_regularizer=regularizers.l2(0.01),
                                    u_regularizer=regularizers.l2(0.01))
    asp_embedding = Embedding(input_dim=6, output_dim=50, trainable=True)
    # sentiment_dense = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))
    sentiment_out = Dense(sentiment_class_num, activation='softmax',
                          kernel_regularizer=regularizers.l2(0.1))
    for i in range(aspect_class_num):
        input_aspect = Input(shape=(1,), )
        inputs.append(input_aspect)
        aspect_embed = asp_embedding(input_aspect)
        # reshape to 2d
        aspect_embed = Flatten()(aspect_embed)
        # repeat aspect for every word in sequence
        repeat_aspect = RepeatVector(max_seq_len)(aspect_embed)

        # mask after concatenate will be same as hidden_out's mask
        concat = concatenate([hidden_vecs_sentiment, repeat_aspect], axis=-1)

        # apply attention mechanism
        attend_weight = sentiment_attention(concat)
        attend_weight_expand = Lambda(lambda x: K.expand_dims(x))(attend_weight)
        attend_hidden = multiply([hidden_vecs_sentiment, attend_weight_expand])
        attend_hidden = Lambda(lambda x: K.sum(x, axis=1))(attend_hidden)

        # aspect_att_and_sentiment_att = Add()([aspect_attentions[i], attend_hidden])
        # aspect_att_and_sentiment_att = attend_hidden
        # aspect_att_and_sentiment_att = Dropout(0.3, name='Dropout_' + str(i))(aspect_att_and_sentiment_att)
        # dense_layer = sentiment_dense(aspect_att_and_sentiment_att)
        # aspect_att_and_sentiment_att = Dropout(0.3, name='Dropout_' + str(i))(aspect_att_and_sentiment_att)
        output_layer = sentiment_out(attend_hidden)
        outputs.append(output_layer)
        loss.append('categorical_crossentropy')
        loss_weights.append(1)

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam()
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    return model


def generate_attention_output_by_attention_weight(hidden_vecs, attention_weight):
    attend_weight_expand = Lambda(lambda x: K.expand_dims(x))(attention_weight)
    attend_hidden = multiply([hidden_vecs, attend_weight_expand])
    result = Lambda(lambda x: K.sum(x, axis=1))(attend_hidden)
    return result


def generate_attention_output_sequence_by_attention_weight(hidden_vecs, attention_weight):
    attend_weight_expand = Lambda(lambda x: K.expand_dims(x))(attention_weight)
    attend_hidden = multiply([hidden_vecs, attend_weight_expand])
    return attend_hidden


def ortho_reg(weight_matrix):
    ### orthogonal regularization for aspect embedding matrix ###
    w_n = weight_matrix / K.cast(
        K.epsilon() + K.sqrt(K.sum(K.square(weight_matrix), axis=-1, keepdims=True)),
        K.floatx())
    reg = K.sum(K.square(K.dot(w_n, K.transpose(w_n)) - K.eye(w_n.shape.as_list()[0])))
    # reg = K.sum(K.square(K.dot(w_n, K.transpose(w_n)) - K.eye(6)))
    return reg


def joint_model_of_aspect_and_sentiment_semeval2014_tan(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding')(word_input)

    # aspect category分类
    text_embed = SpatialDropout1D(0.5)(word_embedding)
    # hidden_vecs = GRU(50, return_sequences=True)(text_embed)
    hidden_vecs, last_state_0, last_state_1 = Bidirectional(GRU(128, return_sequences=True,
                                                           return_state=True,
                                                           name='gru',
                                                           # kernel_regularizer=regularizers.l2(0.01),
                                                           # bias_regularizer=regularizers.l2(0.01)
                                                                ))(text_embed)
    # hidden_vecs = SpatialDropout1D(0.5)(hidden_vecs)
    # hidden_vecs = Dropout(0.3, name='Dropout')(hidden_vecs)
    aspect_attentions = []
    squash_denses = []
    attend_weights = TopicAttention(use_W=False, u_regularizer=ortho_reg, topic_num=6)(hidden_vecs)
    for i in range(len(attend_weights)):
        # apply attention mechanism
        attend_weight = attend_weights[i]
        aspect_att_i = generate_attention_output_by_attention_weight(hidden_vecs, attend_weight)

        aspect_attentions.append(aspect_att_i)
        dense = Dense(16, activation='relu')(aspect_att_i)
        squash_dense = Lambda(lambda x: capsulelayers.squash(x))(dense)
        squash_denses.append(squash_dense)

    squash_denses_concat = concatenate(squash_denses)
    for i in range(aspect_class_num):
        aspect_output = Dense(32, activation="relu")(squash_denses_concat)
        squash_aspect_output = Lambda(lambda x: capsulelayers.squash(x))(aspect_output)
        squash_aspect_output = Dropout(0.6)(squash_aspect_output)
        final_output = Lambda(lambda x: K.sqrt(K.sum(K.square(x), -1, keepdims=True) + K.epsilon()))(squash_aspect_output)
        outputs.append(final_output)
        loss.append('mean_squared_error')
        loss_weights.append(1)

    # 主题对应的情感分类
    hidden_vecs_sentiment, last_state_0_sentiment, last_state_1_sentiment = Bidirectional(GRU(25, return_sequences=True,
                                                                return_state=True,
                                                                name='hidden_vecs_sentiment',
                                                                # kernel_regularizer=regularizers.l2(0.01),
                                                                # bias_regularizer=regularizers.l2(0.01)
                                                                ))(word_embedding)
    hidden_vecs_sentiment = SpatialDropout1D(0.5, name='SpatialDropout1D_sentiment')(hidden_vecs_sentiment)
    sentiment_attention = Attention(W_regularizer=regularizers.l2(0.01),
                                    u_regularizer=regularizers.l2(0.01))
    asp_embedding = Embedding(input_dim=6, output_dim=50, trainable=True)
    # sentiment_dense = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))
    sentiment_out = Dense(sentiment_class_num, activation='softmax',
                          kernel_regularizer=regularizers.l2(0.1))
    for i in range(aspect_class_num):
        input_aspect = Input(shape=(1,), )
        inputs.append(input_aspect)
        aspect_embed = asp_embedding(input_aspect)
        # reshape to 2d
        aspect_embed = Flatten()(aspect_embed)
        # repeat aspect for every word in sequence
        repeat_aspect = RepeatVector(max_seq_len)(aspect_embed)

        # mask after concatenate will be same as hidden_out's mask
        concat = concatenate([hidden_vecs_sentiment, repeat_aspect], axis=-1)

        # apply attention mechanism
        attend_weight = sentiment_attention(concat)
        attend_weight_expand = Lambda(lambda x: K.expand_dims(x))(attend_weight)
        attend_hidden = multiply([hidden_vecs_sentiment, attend_weight_expand])
        attend_hidden = Lambda(lambda x: K.sum(x, axis=1))(attend_hidden)

        # aspect_att_and_sentiment_att = Add()([aspect_attentions[i], attend_hidden])
        # aspect_att_and_sentiment_att = attend_hidden
        # aspect_att_and_sentiment_att = Dropout(0.3, name='Dropout_' + str(i))(aspect_att_and_sentiment_att)
        # dense_layer = sentiment_dense(aspect_att_and_sentiment_att)
        # aspect_att_and_sentiment_att = Dropout(0.3, name='Dropout_' + str(i))(aspect_att_and_sentiment_att)
        output_layer = sentiment_out(attend_hidden)
        outputs.append(output_layer)
        loss.append('categorical_crossentropy')
        loss_weights.append(1)

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam()
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    return model


def joint_model_of_aspect_and_sentiment_semeval2014_old(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding')(word_input)

    # aspect category分类
    text_embed = SpatialDropout1D(0.5)(word_embedding)
    # hidden_vecs = GRU(50, return_sequences=True)(text_embed)
    hidden_vecs, last_state_0, last_state_1 = Bidirectional(GRU(25, return_sequences=True,
                                                           return_state=True,
                                                           name='gru',
                                                           # kernel_regularizer=regularizers.l2(0.01),
                                                           # bias_regularizer=regularizers.l2(0.01)
                                                                ))(text_embed)
    # hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D')(hidden_vecs)
    # hidden_vecs = Dropout(0.3, name='Dropout')(hidden_vecs)
    aspect_attentions = []
    for i in range(aspect_class_num):
        # apply attention mechanism
        attend_weight = Attention(use_W=True, W_regularizer=regularizers.l2(0.01),
                                    u_regularizer=regularizers.l2(0.01))(hidden_vecs)
        aspect_att_i = generate_attention_output_by_attention_weight(hidden_vecs, attend_weight)

        # aspect_att_i = keras_layers.AttLayer(20, i)(hidden_vecs)
        aspect_attentions.append(aspect_att_i)
        final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(aspect_att_i)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01))(final_output)
        outputs.append(aspect_output)
        loss.append('binary_crossentropy')
        loss_weights.append(1)

    # 主题对应的情感分类
    hidden_vecs_sentiment, last_state_0_sentiment, last_state_1_sentiment = Bidirectional(GRU(25, return_sequences=True,
                                                                return_state=True,
                                                                name='hidden_vecs_sentiment',
                                                                # kernel_regularizer=regularizers.l2(0.01),
                                                                # bias_regularizer=regularizers.l2(0.01)
                                                                ))(word_embedding)
    hidden_vecs_sentiment = SpatialDropout1D(0.5, name='SpatialDropout1D_sentiment')(hidden_vecs_sentiment)
    sentiment_attention = Attention(W_regularizer=regularizers.l2(0.01),
                                    u_regularizer=regularizers.l2(0.01))
    asp_embedding = Embedding(input_dim=aspect_class_num + 1, output_dim=50, trainable=True)
    # sentiment_dense = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))
    sentiment_out = Dense(sentiment_class_num, activation='softmax',
                          kernel_regularizer=regularizers.l2(0.1))
    for i in range(aspect_class_num):
        input_aspect = Input(shape=(1,), )
        inputs.append(input_aspect)
        aspect_embed = asp_embedding(input_aspect)
        # reshape to 2d
        aspect_embed = Flatten()(aspect_embed)
        # repeat aspect for every word in sequence
        repeat_aspect = RepeatVector(max_seq_len)(aspect_embed)

        # mask after concatenate will be same as hidden_out's mask
        concat = concatenate([hidden_vecs_sentiment, repeat_aspect], axis=-1)

        # apply attention mechanism
        attend_weight = sentiment_attention(concat)
        attend_weight_expand = Lambda(lambda x: K.expand_dims(x))(attend_weight)
        attend_hidden = multiply([hidden_vecs_sentiment, attend_weight_expand])
        attend_hidden = Lambda(lambda x: K.sum(x, axis=1))(attend_hidden)

        # aspect_att_and_sentiment_att = Add()([aspect_attentions[i], attend_hidden])
        # aspect_att_and_sentiment_att = attend_hidden
        # aspect_att_and_sentiment_att = Dropout(0.3, name='Dropout_' + str(i))(aspect_att_and_sentiment_att)
        # dense_layer = sentiment_dense(aspect_att_and_sentiment_att)
        # aspect_att_and_sentiment_att = Dropout(0.3, name='Dropout_' + str(i))(aspect_att_and_sentiment_att)
        output_layer = sentiment_out(attend_hidden)
        outputs.append(output_layer)
        loss.append('categorical_crossentropy')
        loss_weights.append(1)

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam()
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    return model


def ac_aoa_wrapper(W_dense=None):
    def ac_aoa_inner(inputs):
        """

        :param query:
        :param key:
        :return:
        """
        query, key = inputs
        if W_dense:
            query = W_dense(query)
        dot_result = K.batch_dot(query, key, axes=[1, 2])
        # dot_result = Lambda(lambda x: K.tanh(x))(dot_result)
        attention_weight = K.softmax(dot_result)
        return attention_weight
    return ac_aoa_inner


def ac_aoa(inputs):
    """

    :param query:
    :param key:
    :return:
    """
    query, key = inputs
    dot_result = K.batch_dot(query, key, axes=[1, 2])
    # dot_result = Lambda(lambda x: K.tanh(x))(dot_result)
    attention_weight = K.softmax(dot_result)
    return attention_weight


def joint_model_of_aspect_and_sentiment_semeval2014_acaoa(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding', mask_zero=True)(word_input)

    # aspect category分类
    text_embed = SpatialDropout1D(0.5, name='SpatialDropout1D_embedding')(word_embedding)
    # hidden_vecs = GRU(50, return_sequences=True)(text_embed)
    hidden_vecs, last_state_0, last_state_1 = Bidirectional(GRU(150, return_sequences=True,
                                                           return_state=True,
                                                           name='gru',
                                                           # kernel_regularizer=regularizers.l2(0.01),
                                                           # bias_regularizer=regularizers.l2(0.01)
                                                                ))(text_embed)
    hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D_rnn')(hidden_vecs)
    # hidden_vecs = Dropout(0.3, name='Dropout')(hidden_vecs)
    aspect_attentions = []
    aspect_outputs = []
    for i in range(aspect_class_num):
        # apply attention mechanism
        attend_weight = Attention(W_regularizer=regularizers.l2(0.01),
                                    u_regularizer=regularizers.l2(0.01))(hidden_vecs)
        attend_weight_expand = Lambda(lambda x: K.expand_dims(x))(attend_weight)
        attend_hidden = multiply([hidden_vecs, attend_weight_expand])
        aspect_att_i = Lambda(lambda x: K.sum(x, axis=1))(attend_hidden)

        # aspect_att_i = keras_layers.AttLayer(20, i)(hidden_vecs)
        aspect_attentions.append(aspect_att_i)
        final_output = Dense(32, activation='relu')(aspect_att_i)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01))(final_output)
        aspect_outputs.append(aspect_output)
        outputs.append(aspect_output)
        loss.append('binary_crossentropy')
        loss_weights.append(1.0)

    # 主题对应的情感分类
    hidden_vecs_sentiment, last_state_0_sentiment, last_state_1_sentiment = Bidirectional(GRU(150, return_sequences=True,
                                                                return_state=True,
                                                                name='hidden_vecs_sentiment',
                                                                # kernel_regularizer=regularizers.l2(0.01),
                                                                # bias_regularizer=regularizers.l2(0.01)
                                                                ))(word_embedding)
    hidden_vecs_sentiment = SpatialDropout1D(0.5, name='SpatialDropout1D_sentiment')(hidden_vecs_sentiment)
    # hidden_vecs_sentiment = hidden_vecs
    sentiment_dense = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))
    sentiment_out = Dense(sentiment_class_num, activation='softmax',
                          kernel_regularizer=regularizers.l2(0.1))
    for i in range(aspect_class_num):
        aspect_attention = aspect_attentions[i]
        attend_weight = Lambda(ac_aoa)([aspect_attention, hidden_vecs_sentiment])

        attend_weight_expand = Lambda(lambda x: K.expand_dims(x))(attend_weight)
        attend_hidden = multiply([hidden_vecs_sentiment, attend_weight_expand])
        attend_hidden = Lambda(lambda x: K.sum(x, axis=1))(attend_hidden)
        sentiment_dense_output = sentiment_dense(attend_hidden)
        output_layer = sentiment_out(sentiment_dense_output)
        aspect_output = aspect_outputs[i]
        weight_sentiment_output = multiply([output_layer, aspect_output])
        outputs.append(weight_sentiment_output)
        loss.append('categorical_crossentropy')
        loss_weights.append(1)

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam()
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    return model


def acaoa_semeval_2014_task_4_rest(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding', mask_zero=True)(word_input)

    # aspect category分类
    text_embed = SpatialDropout1D(0.5, name='SpatialDropout1D_embedding')(word_embedding)
    hidden_vecs= Bidirectional(LSTM(100, return_sequences=True,
                                                           return_state=False,
                                                           name='gru',
                                                           # kernel_regularizer=regularizers.l2(0.01),
                                                           # bias_regularizer=regularizers.l2(0.01)
                                                                ))(text_embed)
    hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D_rnn')(hidden_vecs)
    aspect_attentions = []
    for i in range(aspect_class_num):
        # apply attention mechanism
        attend_weight = Attention(use_W=False, W_regularizer=regularizers.l2(0.01),
                                    u_regularizer=regularizers.l2(0.01))(hidden_vecs)
        aspect_att_i = generate_attention_output_by_attention_weight(hidden_vecs, attend_weight)

        aspect_attentions.append(aspect_att_i)
        final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(aspect_att_i)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(final_output)

        outputs.append(aspect_output)
        loss.append(losses.binary_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[0])

    hidden_vecs_sentiment = hidden_vecs
    sentiment_dense = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                            bias_regularizer=regularizers.l2(0.01))
    sentiment_out = Dense(sentiment_class_num, activation='softmax',
                          kernel_regularizer=regularizers.l2(0.1),
                          bias_regularizer=regularizers.l2(0.01))
    for i in range(aspect_class_num):
        aspect_attention = aspect_attentions[i]
        attend_weight = Lambda(ac_aoa)([aspect_attention, hidden_vecs_sentiment])
        attend_hidden = generate_attention_output_by_attention_weight(hidden_vecs_sentiment, attend_weight)

        sentiment_dense_output = sentiment_dense(attend_hidden)
        output_layer = sentiment_out(sentiment_dense_output)

        outputs.append(output_layer)
        loss.append(losses.categorical_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[1])

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file=data_path.data_base_dir + 'semeval_2014_task_4_rest_acaoa.png')
    return model


def acaoa_semeval_2016_task_5_ch_came_sb1(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding', mask_zero=True)(word_input)

    # aspect category分类
    text_embed = SpatialDropout1D(0.5, name='SpatialDropout1D_embedding')(word_embedding)
    hidden_vecs= Bidirectional(LSTM(100, return_sequences=True,
                                                           return_state=False,
                                                           name='gru',
                                                           # kernel_regularizer=regularizers.l2(0.01),
                                                           # bias_regularizer=regularizers.l2(0.01)
                                                                ))(text_embed)
    hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D_rnn')(hidden_vecs)
    aspect_attentions = []
    for i in range(aspect_class_num):
        # apply attention mechanism
        attend_weight = Attention(use_W=False, W_regularizer=regularizers.l2(0.01),
                                    u_regularizer=regularizers.l2(0.01))(hidden_vecs)
        aspect_att_i = generate_attention_output_by_attention_weight(hidden_vecs, attend_weight)

        aspect_attentions.append(aspect_att_i)
        final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(aspect_att_i)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(final_output)

        outputs.append(aspect_output)
        loss.append(losses.binary_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[0])

    hidden_vecs_sentiment = hidden_vecs
    sentiment_dense = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                            bias_regularizer=regularizers.l2(0.01))
    sentiment_out = Dense(sentiment_class_num, activation='softmax',
                          kernel_regularizer=regularizers.l2(0.1),
                          bias_regularizer=regularizers.l2(0.01))
    for i in range(aspect_class_num):
        aspect_attention = aspect_attentions[i]
        attend_weight = Lambda(ac_aoa)([aspect_attention, hidden_vecs_sentiment])
        attend_hidden = generate_attention_output_by_attention_weight(hidden_vecs_sentiment, attend_weight)

        sentiment_dense_output = sentiment_dense(attend_hidden)
        output_layer = sentiment_out(sentiment_dense_output)

        outputs.append(output_layer)
        loss.append(losses.categorical_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[1])

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file=data_path.data_base_dir + 'semeval_2014_task_4_rest_acaoa.png')
    return model


def acaoa_semeval_2016_task_5_rest_sb1(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding', mask_zero=True)(word_input)

    # aspect category分类
    text_embed = SpatialDropout1D(0.5, name='SpatialDropout1D_embedding')(word_embedding)
    hidden_vecs= Bidirectional(LSTM(100, return_sequences=True,
                                                           return_state=False,
                                                           name='gru',
                                                           # kernel_regularizer=regularizers.l2(0.01),
                                                           # bias_regularizer=regularizers.l2(0.01)
                                                                ))(text_embed)
    hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D_rnn')(hidden_vecs)
    aspect_attentions = []
    for i in range(aspect_class_num):
        # apply attention mechanism
        attend_weight = Attention(use_W=False, W_regularizer=regularizers.l2(0.01),
                                    u_regularizer=regularizers.l2(0.01))(hidden_vecs)
        aspect_att_i = generate_attention_output_by_attention_weight(hidden_vecs, attend_weight)

        aspect_attentions.append(aspect_att_i)
        final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(aspect_att_i)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(final_output)

        outputs.append(aspect_output)
        loss.append(losses.binary_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[0])

    hidden_vecs_sentiment = hidden_vecs
    sentiment_dense = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                            bias_regularizer=regularizers.l2(0.01))
    sentiment_out = Dense(sentiment_class_num, activation='softmax',
                          kernel_regularizer=regularizers.l2(0.1),
                          bias_regularizer=regularizers.l2(0.01))
    for i in range(aspect_class_num):
        aspect_attention = aspect_attentions[i]
        attend_weight = Lambda(ac_aoa)([aspect_attention, hidden_vecs_sentiment])
        attend_hidden = generate_attention_output_by_attention_weight(hidden_vecs_sentiment, attend_weight)

        sentiment_dense_output = sentiment_dense(attend_hidden)
        output_layer = sentiment_out(sentiment_dense_output)

        outputs.append(output_layer)
        loss.append(losses.categorical_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[1])

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file=data_path.data_base_dir + 'semeval_2014_task_4_rest_acaoa.png')
    return model


def acaoa_semeval_2016_task_5_rest_sb2(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding', mask_zero=True)(word_input)

    # aspect category分类
    text_embed = SpatialDropout1D(0.5, name='SpatialDropout1D_embedding')(word_embedding)
    hidden_vecs= Bidirectional(LSTM(100, return_sequences=True,
                                                           return_state=False,
                                                           name='gru',
                                                           # kernel_regularizer=regularizers.l2(0.01),
                                                           # bias_regularizer=regularizers.l2(0.01)
                                                                ))(text_embed)
    hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D_rnn')(hidden_vecs)
    aspect_attentions = []
    for i in range(aspect_class_num):
        # apply attention mechanism
        attend_weight = Attention(use_W=False, W_regularizer=regularizers.l2(0.01),
                                    u_regularizer=regularizers.l2(0.01))(hidden_vecs)
        aspect_att_i = generate_attention_output_by_attention_weight(hidden_vecs, attend_weight)

        aspect_attentions.append(aspect_att_i)
        final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(aspect_att_i)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(final_output)

        outputs.append(aspect_output)
        loss.append(losses.binary_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[0])

    hidden_vecs_sentiment = hidden_vecs
    sentiment_dense = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                            bias_regularizer=regularizers.l2(0.01))
    sentiment_out = Dense(sentiment_class_num, activation='softmax',
                          kernel_regularizer=regularizers.l2(0.1),
                          bias_regularizer=regularizers.l2(0.01))
    for i in range(aspect_class_num):
        aspect_attention = aspect_attentions[i]
        attend_weight = Lambda(ac_aoa)([aspect_attention, hidden_vecs_sentiment])
        attend_hidden = generate_attention_output_by_attention_weight(hidden_vecs_sentiment, attend_weight)

        sentiment_dense_output = sentiment_dense(attend_hidden)
        output_layer = sentiment_out(sentiment_dense_output)

        outputs.append(output_layer)
        loss.append(losses.categorical_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[1])

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file=data_path.data_base_dir + 'semeval_2014_task_4_rest_acaoa.png')
    return model


def acaoa_block_semeval_2016_task_5_lapt_sb2(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    aspect_index_and_attention = {}

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding', mask_zero=True)(word_input)

    # aspect category分类
    text_embed = SpatialDropout1D(0.5, name='SpatialDropout1D_embedding')(word_embedding)

    for i in range(aspect_class_num):
        aspect_att, sentiment_att = aae_aoa_block(text_embed)
        if i not in aspect_index_and_attention:
            aspect_index_and_attention[i] = []
        aspect_index_and_attention[i].append((aspect_att, sentiment_att))

    hidden_vecs= Bidirectional(LSTM(100, return_sequences=True,
                                                           return_state=False,
                                                           name='gru',
                                                           # kernel_regularizer=regularizers.l2(0.01),
                                                           # bias_regularizer=regularizers.l2(0.01)
                                                                ))(text_embed)
    hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D_rnn')(hidden_vecs)

    for i in range(aspect_class_num):
        aspect_att, sentiment_att = aae_aoa_block(hidden_vecs)
        if i not in aspect_index_and_attention:
            aspect_index_and_attention[i] = []
        aspect_index_and_attention[i].append((aspect_att, sentiment_att))

    aspect_attentions = []
    for i in range(aspect_class_num):
        # apply attention mechanism
        i_aspect_attentions = [a[0] for a in aspect_index_and_attention[i]]
        if len(i_aspect_attentions) == 1:
            aspect_att_i = i_aspect_attentions[0]
        else:
            aspect_att_i = concatenate(i_aspect_attentions)

        aspect_attentions.append(aspect_att_i)
        final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(aspect_att_i)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(final_output)

        outputs.append(aspect_output)
        loss.append(losses.binary_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[0])

    sentiment_dense = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                            bias_regularizer=regularizers.l2(0.01))
    sentiment_out = Dense(sentiment_class_num, activation='softmax',
                          kernel_regularizer=regularizers.l2(0.1),
                          bias_regularizer=regularizers.l2(0.01))
    for i in range(aspect_class_num):
        i_sentiment_attentions = [a[1] for a in aspect_index_and_attention[i]]
        if len(i_sentiment_attentions) == 1:
            attend_hidden = i_aspect_attentions[0]
        else:
            attend_hidden = concatenate(i_aspect_attentions)

        sentiment_dense_output = sentiment_dense(attend_hidden)
        output_layer = sentiment_out(sentiment_dense_output)

        outputs.append(output_layer)
        loss.append(losses.categorical_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[1])

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file=data_path.data_base_dir + 'semeval_2014_task_4_rest_acaoa.png')
    return model


def aae_without_share(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    aspect_index_and_attention = {}

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding', mask_zero=True)(word_input)

    # aspect category分类
    text_embed = SpatialDropout1D(0.5, name='SpatialDropout1D_embedding')(word_embedding)

    for i in range(aspect_class_num):
        aspect_att, sentiment_att = aae_aoa_block(text_embed)
        if i not in aspect_index_and_attention:
            aspect_index_and_attention[i] = []
        aspect_index_and_attention[i].append((aspect_att, sentiment_att))

    hidden_vecs= Bidirectional(LSTM(100, return_sequences=True,
                                                           return_state=False,
                                                           name='gru',
                                                           # kernel_regularizer=regularizers.l2(0.01),
                                                           # bias_regularizer=regularizers.l2(0.01)
                                                                ))(text_embed)
    hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D_rnn')(hidden_vecs)

    for i in range(aspect_class_num):
        aspect_att, sentiment_att = aae_aoa_block(hidden_vecs)
        if i not in aspect_index_and_attention:
            aspect_index_and_attention[i] = []
        aspect_index_and_attention[i].append((aspect_att, sentiment_att))

    aspect_attentions = []
    for i in range(aspect_class_num):
        # apply attention mechanism
        i_aspect_attentions = [a[0] for a in aspect_index_and_attention[i]]
        if len(i_aspect_attentions) == 1:
            aspect_att_i = i_aspect_attentions[0]
        else:
            aspect_att_i = concatenate(i_aspect_attentions)

        aspect_attentions.append(aspect_att_i)
        final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(aspect_att_i)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(final_output)

        outputs.append(aspect_output)
        loss.append(losses.binary_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[0])

    for i in range(aspect_class_num):
        i_sentiment_attentions = [a[1] for a in aspect_index_and_attention[i]]
        if len(i_sentiment_attentions) == 1:
            attend_hidden = i_sentiment_attentions[0]
        else:
            attend_hidden = concatenate(i_sentiment_attentions)

        sentiment_dense_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                            bias_regularizer=regularizers.l2(0.01))(attend_hidden)
        output_layer = Dense(sentiment_class_num, activation='softmax',
                          kernel_regularizer=regularizers.l2(0.1),
                          bias_regularizer=regularizers.l2(0.01))(sentiment_dense_output)

        outputs.append(output_layer)
        loss.append(losses.categorical_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[1])

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file=data_path.data_base_dir + 'semeval_2014_task_4_rest_acaoa.png')
    return model



def acaoa_block_semeval_2016_task_5_ch_came_sb1(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    aspect_index_and_attention = {}

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding', mask_zero=True)(word_input)

    # aspect category分类
    text_embed = SpatialDropout1D(0.5, name='SpatialDropout1D_embedding')(word_embedding)

    for i in range(aspect_class_num):
        aspect_att, sentiment_att = aae_aoa_block(text_embed)
        if i not in aspect_index_and_attention:
            aspect_index_and_attention[i] = []
        aspect_index_and_attention[i].append((aspect_att, sentiment_att))

    hidden_vecs= Bidirectional(LSTM(100, return_sequences=True,
                                                           return_state=False,
                                                           name='gru',
                                                           # kernel_regularizer=regularizers.l2(0.01),
                                                           # bias_regularizer=regularizers.l2(0.01)
                                                                ))(text_embed)
    hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D_rnn')(hidden_vecs)

    for i in range(aspect_class_num):
        aspect_att, sentiment_att = aae_aoa_block(hidden_vecs)
        if i not in aspect_index_and_attention:
            aspect_index_and_attention[i] = []
        aspect_index_and_attention[i].append((aspect_att, sentiment_att))

    aspect_attentions = []
    for i in range(aspect_class_num):
        # apply attention mechanism
        i_aspect_attentions = [a[0] for a in aspect_index_and_attention[i]]
        if len(i_aspect_attentions) == 1:
            aspect_att_i = i_aspect_attentions[0]
        else:
            aspect_att_i = concatenate(i_aspect_attentions)

        aspect_attentions.append(aspect_att_i)
        final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(aspect_att_i)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(final_output)

        outputs.append(aspect_output)
        loss.append(losses.binary_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[0])

    sentiment_dense = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                            bias_regularizer=regularizers.l2(0.01))
    sentiment_out = Dense(sentiment_class_num, activation='softmax',
                          kernel_regularizer=regularizers.l2(0.1),
                          bias_regularizer=regularizers.l2(0.01))
    for i in range(aspect_class_num):
        i_sentiment_attentions = [a[1] for a in aspect_index_and_attention[i]]
        if len(i_sentiment_attentions) == 1:
            attend_hidden = i_aspect_attentions[0]
        else:
            attend_hidden = concatenate(i_aspect_attentions)

        sentiment_dense_output = sentiment_dense(attend_hidden)
        output_layer = sentiment_out(sentiment_dense_output)

        outputs.append(output_layer)
        loss.append(losses.categorical_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[1])

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file=data_path.data_base_dir + 'semeval_2014_task_4_rest_acaoa.png')
    return model


def acaoa_block_semeval_2016_task_5_ch_phns_sb1(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    aspect_index_and_attention = {}

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding', mask_zero=True)(word_input)

    # aspect category分类
    text_embed = SpatialDropout1D(0.5, name='SpatialDropout1D_embedding')(word_embedding)

    for i in range(aspect_class_num):
        aspect_att, sentiment_att = aae_aoa_block(text_embed)
        if i not in aspect_index_and_attention:
            aspect_index_and_attention[i] = []
        aspect_index_and_attention[i].append((aspect_att, sentiment_att))

    hidden_vecs= Bidirectional(LSTM(100, return_sequences=True,
                                                           return_state=False,
                                                           name='gru',
                                                           # kernel_regularizer=regularizers.l2(0.01),
                                                           # bias_regularizer=regularizers.l2(0.01)
                                                                ))(text_embed)
    hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D_rnn')(hidden_vecs)

    for i in range(aspect_class_num):
        aspect_att, sentiment_att = aae_aoa_block(hidden_vecs)
        if i not in aspect_index_and_attention:
            aspect_index_and_attention[i] = []
        aspect_index_and_attention[i].append((aspect_att, sentiment_att))

    aspect_attentions = []
    for i in range(aspect_class_num):
        # apply attention mechanism
        i_aspect_attentions = [a[0] for a in aspect_index_and_attention[i]]
        if len(i_aspect_attentions) == 1:
            aspect_att_i = i_aspect_attentions[0]
        else:
            aspect_att_i = concatenate(i_aspect_attentions)

        aspect_attentions.append(aspect_att_i)
        final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(aspect_att_i)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(final_output)

        outputs.append(aspect_output)
        loss.append(losses.binary_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[0])

    sentiment_dense = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                            bias_regularizer=regularizers.l2(0.01))
    sentiment_out = Dense(sentiment_class_num, activation='softmax',
                          kernel_regularizer=regularizers.l2(0.1),
                          bias_regularizer=regularizers.l2(0.01))
    for i in range(aspect_class_num):
        i_sentiment_attentions = [a[1] for a in aspect_index_and_attention[i]]
        if len(i_sentiment_attentions) == 1:
            attend_hidden = i_aspect_attentions[0]
        else:
            attend_hidden = concatenate(i_aspect_attentions)

        sentiment_dense_output = sentiment_dense(attend_hidden)
        output_layer = sentiment_out(sentiment_dense_output)

        outputs.append(output_layer)
        loss.append(losses.categorical_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[1])

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file=data_path.data_base_dir + 'semeval_2014_task_4_rest_acaoa.png')
    return model


def cae(max_seq_len, input_dim, output_dim, embedding_matrix, aspect_class_num=10, sentiment_class_num=3,
        aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    aspect_index_and_attention = {}

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding', mask_zero=True)(word_input)

    # aspect category分类
    text_embed = SpatialDropout1D(0.5, name='SpatialDropout1D_embedding')(word_embedding)

    for i in range(aspect_class_num):
        aspect_att, sentiment_att = aae_aoa_block(text_embed, layer_index=0, aspect_index=i)
        if i not in aspect_index_and_attention:
            aspect_index_and_attention[i] = []
        aspect_index_and_attention[i].append((aspect_att, sentiment_att))

    hidden_vecs= Bidirectional(LSTM(100, return_sequences=True,
                                                           return_state=False,
                                                           name='gru',
                                                           # kernel_regularizer=regularizers.l2(0.01),
                                                           # bias_regularizer=regularizers.l2(0.01)
                                                                ))(text_embed)
    hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D_rnn')(hidden_vecs)

    for i in range(aspect_class_num):
        aspect_att, sentiment_att = aae_aoa_block(hidden_vecs, layer_index=1, aspect_index=i)
        if i not in aspect_index_and_attention:
            aspect_index_and_attention[i] = []
        aspect_index_and_attention[i].append((aspect_att, sentiment_att))

    aspect_attentions = []
    for i in range(aspect_class_num):
        # apply attention mechanism
        i_aspect_attentions = [a[0] for a in aspect_index_and_attention[i]]
        if len(i_aspect_attentions) == 1:
            aspect_att_i = i_aspect_attentions[0]
        else:
            aspect_att_i = concatenate(i_aspect_attentions)

        aspect_attentions.append(aspect_att_i)
        final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(aspect_att_i)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(final_output)

        outputs.append(aspect_output)
        loss.append(losses.binary_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[0])

    sentiment_dense = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                            bias_regularizer=regularizers.l2(0.01))
    sentiment_out = Dense(sentiment_class_num, activation='softmax',
                          kernel_regularizer=regularizers.l2(0.1),
                          bias_regularizer=regularizers.l2(0.01))
    for i in range(aspect_class_num):
        i_sentiment_attentions = [a[1] for a in aspect_index_and_attention[i]]
        if len(i_sentiment_attentions) == 1:
            attend_hidden = i_sentiment_attentions[0]
        else:
            attend_hidden = concatenate(i_sentiment_attentions)

        sentiment_dense_output = sentiment_dense(attend_hidden)
        output_layer = sentiment_out(sentiment_dense_output)

        outputs.append(output_layer)
        loss.append(losses.categorical_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[1])

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    return model


def cae2(max_seq_len, input_dim, output_dim, embedding_matrix, aspect_class_num=10, sentiment_class_num=3,
        aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    aspect_index_and_attention = {}

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding', mask_zero=True)(word_input)

    # aspect category分类
    text_embed = SpatialDropout1D(0.5, name='SpatialDropout1D_embedding')(word_embedding)

    for i in range(aspect_class_num):
        aspect_att, sentiment_att = aae_aoa_block(text_embed, layer_index=0, aspect_index=i)
        if i not in aspect_index_and_attention:
            aspect_index_and_attention[i] = []
        aspect_index_and_attention[i].append((aspect_att, sentiment_att))

    hidden_vecs= Bidirectional(LSTM(100, return_sequences=True,
                                                           return_state=False,
                                                           name='gru',
                                                           # kernel_regularizer=regularizers.l2(0.01),
                                                           # bias_regularizer=regularizers.l2(0.01)
                                                                ))(text_embed)
    hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D_rnn')(hidden_vecs)

    for i in range(aspect_class_num):
        aspect_att, sentiment_att = aae_aoa_block(hidden_vecs, layer_index=1, aspect_index=i)
        if i not in aspect_index_and_attention:
            aspect_index_and_attention[i] = []
        aspect_index_and_attention[i].append((aspect_att, sentiment_att))

    aspect_attentions = []
    for i in range(aspect_class_num):
        # apply attention mechanism
        i_aspect_attentions = [a[0] for a in aspect_index_and_attention[i]]
        if len(i_aspect_attentions) == 1:
            aspect_att_i = i_aspect_attentions[0]
        else:
            aspect_att_i = concatenate(i_aspect_attentions)

        aspect_attentions.append(aspect_att_i)
        final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(aspect_att_i)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(final_output)

        outputs.append(aspect_output)
        loss.append(losses.binary_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[0])

    sentiment_dense = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                            bias_regularizer=regularizers.l2(0.01))
    sentiment_out = Dense(sentiment_class_num, activation='softmax',
                          kernel_regularizer=regularizers.l2(0.1),
                          bias_regularizer=regularizers.l2(0.01))
    for i in range(aspect_class_num):
        i_sentiment_attentions = [a[1] for a in aspect_index_and_attention[i]]
        if len(i_sentiment_attentions) == 1:
            attend_hidden = i_sentiment_attentions[0]
        else:
            attend_hidden = concatenate(i_sentiment_attentions)

        sentiment_dense_output = sentiment_dense(attend_hidden)
        output_layer = sentiment_out(sentiment_dense_output)

        outputs.append(output_layer)
        loss.append(losses.categorical_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[1])

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    return model


def acd_attention(max_seq_len, input_dim, output_dim, embedding_matrix, aspect_class_num=10):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    aspect_index_and_attention = {}

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding', mask_zero=True)(word_input)

    # aspect category分类
    text_embed = SpatialDropout1D(0.5, name='SpatialDropout1D_embedding')(word_embedding)

    hidden_vecs = Bidirectional(LSTM(100, return_sequences=True,
                                     return_state=False,
                                     name='gru',
                                     # kernel_regularizer=regularizers.l2(0.01),
                                     # bias_regularizer=regularizers.l2(0.01)
                                     ))(text_embed)
    hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D_rnn')(hidden_vecs)

    for i in range(aspect_class_num):
        attend_weight = Attention(use_W=True, use_bias=False, u_size=5, W_regularizer=regularizers.l2(0.01),
                                  u_regularizer=regularizers.l2(0.01),
                                  name='attention_%d' % i)(hidden_vecs)
        aspect_att = generate_attention_output_by_attention_weight(hidden_vecs, attend_weight)

        final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                             bias_regularizer=regularizers.l2(0.01))(aspect_att)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01),
                              bias_regularizer=regularizers.l2(0.01))(final_output)
        outputs.append(aspect_output)
        loss.append(losses.binary_crossentropy)
        loss_weights.append(1)

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    return model


class AcdTwoStageModel:
    """
    recognize entity firstly, then recognize aspect based on entity information
    """

    def entity_classifier(self, hidden_vectors, entity_index):
        attend_weight = Attention(use_W=True, use_bias=False, u_size=5, W_regularizer=regularizers.l2(0.01),
                                  u_regularizer=regularizers.l2(0.01),
                                  name='entity_attention_%d' % entity_index)(hidden_vectors)
        entity_att = generate_attention_output_by_attention_weight(hidden_vectors, attend_weight)

        final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                             bias_regularizer=regularizers.l2(0.01))(entity_att)
        entity_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01),
                              bias_regularizer=regularizers.l2(0.01))(final_output)
        return entity_output, attend_weight

    def aspect_classifier(self, hidden_vectors, entity_attention_weight):
        aspect_att = generate_attention_output_by_attention_weight(hidden_vectors, entity_attention_weight)

        final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                             bias_regularizer=regularizers.l2(0.01))(aspect_att)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01),
                              bias_regularizer=regularizers.l2(0.01))(final_output)
        return aspect_output

    def __call__(self, max_seq_len, input_dim, output_dim, embedding_matrix, entities, shared_aspects,
                 entity_aspect_pairs):
        """

        :param max_seq_len:
        :param input_dim:
        :param output_dim:
        :param embedding_matrix:
        :param entities:
        :param shared_aspects:
        :param entity_aspect_pairs:
        :return:
        """

        inputs = []
        outputs = []
        loss = []
        loss_weights = []

        word_input = Input(shape=(max_seq_len,), name='word_input')
        inputs.append(word_input)
        word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                                   name='word_embedding', mask_zero=True)(word_input)

        word_embedding = SpatialDropout1D(0.5, name='SpatialDropout1D_embedding')(word_embedding)

        # entity classification
        entity_and_attention_weight = {}
        entity_outputs = []
        entity_loss = []
        for i, entity in enumerate(entities):
            entity_output, entity_attention_weight = self.entity_classifier(word_embedding, i)
            entity_and_attention_weight[entity] = entity_attention_weight
            entity_outputs.append(entity_output)
            entity_loss.append(losses.binary_crossentropy)

        hidden_vecs = Bidirectional(LSTM(100, return_sequences=True,
                                         return_state=False,
                                         name='gru',
                                         # kernel_regularizer=regularizers.l2(0.01),
                                         # bias_regularizer=regularizers.l2(0.01)
                                         ))(word_embedding)
        hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D_rnn')(hidden_vecs)

        aspect_outputs = []
        aspect_loss = []
        for i, (entity, aspect) in enumerate(entity_aspect_pairs):
            entity_attention_weight = entity_and_attention_weight[entity]
            aspect_output = self.aspect_classifier(hidden_vecs, entity_attention_weight)
            aspect_outputs.append(aspect_output)
            aspect_loss.append(losses.binary_crossentropy)

        outputs.extend(aspect_outputs)
        outputs.extend(entity_outputs)
        loss.extend(aspect_loss)
        loss.extend(entity_loss)
        model = Model(inputs=inputs, outputs=outputs, name='model')
        adam = optimizers.adam(clipvalue=5)
        model.compile(loss=loss,
                      optimizer=adam,
                      # loss_weights=loss_weights,
                      metrics=['accuracy'])
        model.summary()
        return model


class AspectClassifierForAcdTwoStageModel:
    """

    """

    def __init__(self):
        self.dense_layer = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                                 bias_regularizer=regularizers.l2(0.01))
        self.output_layer = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01),
                                  bias_regularizer=regularizers.l2(0.01))

    def __call__(self, hidden_vectors, entity_attention_weight):
        aspect_att = generate_attention_output_by_attention_weight(hidden_vectors, entity_attention_weight)
        final_output = self.dense_layer(aspect_att)
        aspect_output = self.output_layer(final_output)
        return aspect_output


class AcdTwoStageModelWithSharedAspectParameter:
    """
    recognize entity firstly, then recognize aspect based on entity information
    """

    def entity_classifier(self, hidden_vectors, entity_index):
        attend_weight = Attention(use_W=True, use_bias=False, u_size=5, W_regularizer=regularizers.l2(0.01),
                                  u_regularizer=regularizers.l2(0.01),
                                  name='entity_attention_%d' % entity_index)(hidden_vectors)
        entity_att = generate_attention_output_by_attention_weight(hidden_vectors, attend_weight)

        final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                             bias_regularizer=regularizers.l2(0.01))(entity_att)
        entity_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01),
                              bias_regularizer=regularizers.l2(0.01))(final_output)
        return entity_output, attend_weight

    def aspect_classifier(self, hidden_vectors, entity_attention_weight):
        aspect_att = generate_attention_output_by_attention_weight(hidden_vectors, entity_attention_weight)

        final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                             bias_regularizer=regularizers.l2(0.01))(aspect_att)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01),
                              bias_regularizer=regularizers.l2(0.01))(final_output)
        return aspect_output

    def __call__(self, max_seq_len, input_dim, output_dim, embedding_matrix, entities, shared_aspects,
                 entity_aspect_pairs):
        """

        :param max_seq_len:
        :param input_dim:
        :param output_dim:
        :param embedding_matrix:
        :param entities:
        :param shared_aspects:
        :param entity_aspect_pairs:
        :return:
        """

        inputs = []
        outputs = []
        loss = []
        loss_weights = []

        word_input = Input(shape=(max_seq_len,), name='word_input')
        inputs.append(word_input)
        word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                                   name='word_embedding', mask_zero=True)(word_input)

        word_embedding = SpatialDropout1D(0.5, name='SpatialDropout1D_embedding')(word_embedding)

        # entity classification
        entity_and_attention_weight = {}
        entity_outputs = []
        entity_loss = []
        for i, entity in enumerate(entities):
            entity_output, entity_attention_weight = self.entity_classifier(word_embedding, i)
            entity_and_attention_weight[entity] = entity_attention_weight
            entity_outputs.append(entity_output)
            entity_loss.append(losses.binary_crossentropy)

        hidden_vecs = Bidirectional(LSTM(100, return_sequences=True,
                                         return_state=False,
                                         name='gru',
                                         # kernel_regularizer=regularizers.l2(0.01),
                                         # bias_regularizer=regularizers.l2(0.01)
                                         ))(word_embedding)
        hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D_rnn')(hidden_vecs)

        aspect_outputs = []
        aspect_loss = []
        shared_aspect_and_classifier = {}
        for aspect in shared_aspects:
            shared_aspect_and_classifier[aspect] = AspectClassifierForAcdTwoStageModel()
        for i, (entity, aspect) in enumerate(entity_aspect_pairs):
            entity_attention_weight = entity_and_attention_weight[entity]
            if aspect in shared_aspect_and_classifier:
                aspect_classifier = shared_aspect_and_classifier[aspect]
            else:
                aspect_classifier = AspectClassifierForAcdTwoStageModel()
            aspect_output = aspect_classifier(hidden_vecs, entity_attention_weight)
            aspect_outputs.append(aspect_output)
            aspect_loss.append(losses.binary_crossentropy)

        outputs.extend(aspect_outputs)
        outputs.extend(entity_outputs)
        loss.extend(aspect_loss)
        loss.extend(entity_loss)
        model = Model(inputs=inputs, outputs=outputs, name='model')
        adam = optimizers.adam(clipvalue=5)
        model.compile(loss=loss,
                      optimizer=adam,
                      # loss_weights=loss_weights,
                      metrics=['accuracy'])
        model.summary()
        return model


class AcdTwoStageModelWithSharedAspectParameterGcn:
    """
    recognize entity firstly, then recognize aspect based on entity information
    """

    def entity_classifier(self, hidden_vectors, entity_index):
        attend_weight = Attention(use_W=True, use_bias=False, u_size=5, W_regularizer=regularizers.l2(0.01),
                                  u_regularizer=regularizers.l2(0.01),
                                  name='entity_attention_%d' % entity_index)(hidden_vectors)
        entity_att = generate_attention_output_by_attention_weight(hidden_vectors, attend_weight)

        final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                             bias_regularizer=regularizers.l2(0.01))(entity_att)
        entity_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01),
                              bias_regularizer=regularizers.l2(0.01))(final_output)
        return entity_output, attend_weight

    def aspect_classifier(self, hidden_vectors, entity_attention_weight):
        aspect_att = generate_attention_output_by_attention_weight(hidden_vectors, entity_attention_weight)

        final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                             bias_regularizer=regularizers.l2(0.01))(aspect_att)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01),
                              bias_regularizer=regularizers.l2(0.01))(final_output)
        return aspect_output

    def __call__(self, max_seq_len, input_dim, output_dim, embedding_matrix, entities, shared_aspects,
                 entity_aspect_pairs):
        """

        :param max_seq_len:
        :param input_dim:
        :param output_dim:
        :param embedding_matrix:
        :param entities:
        :param shared_aspects:
        :param entity_aspect_pairs:
        :return:
        """

        inputs = []
        outputs = []
        loss = []
        loss_weights = []

        word_input = Input(shape=(max_seq_len,), name='word_input')
        inputs.append(word_input)
        word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                                   name='word_embedding', mask_zero=True)(word_input)

        word_embedding = SpatialDropout1D(0.5, name='SpatialDropout1D_embedding')(word_embedding)

        # entity classification
        entity_and_attention_weight = {}
        entity_outputs = []
        entity_loss = []
        for i, entity in enumerate(entities):
            entity_output, entity_attention_weight = self.entity_classifier(word_embedding, i)
            entity_and_attention_weight[entity] = entity_attention_weight
            entity_outputs.append(entity_output)
            entity_loss.append(losses.binary_crossentropy)

        hidden_vecs = Bidirectional(LSTM(100, return_sequences=True,
                                         return_state=False,
                                         name='gru',
                                         # kernel_regularizer=regularizers.l2(0.01),
                                         # bias_regularizer=regularizers.l2(0.01)
                                         ))(word_embedding)
        hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D_rnn')(hidden_vecs)

        aspect_outputs = []
        aspect_loss = []
        shared_aspect_and_classifier = {}
        for aspect in shared_aspects:
            shared_aspect_and_classifier[aspect] = AspectClassifierForAcdTwoStageModel()
        for i, (entity, aspect) in enumerate(entity_aspect_pairs):
            entity_attention_weight = entity_and_attention_weight[entity]
            if aspect in shared_aspect_and_classifier:
                aspect_classifier = shared_aspect_and_classifier[aspect]
            else:
                aspect_classifier = AspectClassifierForAcdTwoStageModel()
            aspect_output = aspect_classifier(hidden_vecs, entity_attention_weight)
            aspect_outputs.append(aspect_output)
            aspect_loss.append(losses.binary_crossentropy)

        outputs.extend(aspect_outputs)
        outputs.extend(entity_outputs)
        loss.extend(aspect_loss)
        loss.extend(entity_loss)
        model = Model(inputs=inputs, outputs=outputs, name='model')
        adam = optimizers.adam(clipvalue=5)
        model.compile(loss=loss,
                      optimizer=adam,
                      # loss_weights=loss_weights,
                      metrics=['accuracy'])
        model.summary()
        return model


def entity_detection_attention(max_seq_len, input_dim, output_dim, embedding_matrix, aspect_class_num=10):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    aspect_index_and_attention = {}

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding', mask_zero=True)(word_input)

    # aspect category分类
    text_embed = SpatialDropout1D(0.5, name='SpatialDropout1D_embedding')(word_embedding)

    hidden_vecs = Bidirectional(LSTM(100, return_sequences=True,
                                     return_state=False,
                                     name='gru',
                                     # kernel_regularizer=regularizers.l2(0.01),
                                     # bias_regularizer=regularizers.l2(0.01)
                                     ))(text_embed)
    hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D_rnn')(hidden_vecs)

    for i in range(aspect_class_num):
        attend_weight = Attention(use_W=True, use_bias=False, u_size=5, W_regularizer=regularizers.l2(0.01),
                                  u_regularizer=regularizers.l2(0.01),
                                  name='attention_%d' % i)(hidden_vecs)
        aspect_att = generate_attention_output_by_attention_weight(hidden_vecs, attend_weight)

        final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                             bias_regularizer=regularizers.l2(0.01))(aspect_att)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01),
                              bias_regularizer=regularizers.l2(0.01))(final_output)
        outputs.append(aspect_output)
        loss.append(losses.binary_crossentropy)
        loss_weights.append(1)

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    return model


def entity_detection_average_attention(max_seq_len, input_dim, output_dim, embedding_matrix, aspect_class_num=10):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    aspect_index_and_attention = {}

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding', mask_zero=True)(word_input)

    # aspect category分类
    text_embed = SpatialDropout1D(0.5, name='SpatialDropout1D_embedding')(word_embedding)

    hidden_vecs = Bidirectional(LSTM(100, return_sequences=True,
                                     return_state=False,
                                     name='gru',
                                     # kernel_regularizer=regularizers.l2(0.01),
                                     # bias_regularizer=regularizers.l2(0.01)
                                     ))(text_embed)
    hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D_rnn')(hidden_vecs)

    for i in range(aspect_class_num):
        attend_weight = Attention(use_W=True, use_bias=False, u_size=5, W_regularizer=regularizers.l2(0.01),
                                  u_regularizer=regularizers.l2(0.01), just_average=True,
                                  name='attention_%d' % i)(hidden_vecs)
        aspect_att = generate_attention_output_by_attention_weight(hidden_vecs, attend_weight)

        final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                             bias_regularizer=regularizers.l2(0.01))(aspect_att)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01),
                              bias_regularizer=regularizers.l2(0.01))(final_output)
        outputs.append(aspect_output)
        loss.append(losses.binary_crossentropy)
        loss_weights.append(1)

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    return model


def slsa_attention(max_seq_len, input_dim, output_dim, embedding_matrix, polarity_class_num=3):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding', mask_zero=True)(word_input)

    # aspect category分类
    text_embed = SpatialDropout1D(0.5, name='SpatialDropout1D_embedding')(word_embedding)

    attend_weight = Attention(use_W=True, use_bias=False, u_size=5, W_regularizer=regularizers.l2(0.01),
                              u_regularizer=regularizers.l2(0.01))(text_embed)
    if task_conf.log_attention:
        attend_weight = log_tensor(attend_weight, '\nac_%d:' % 1)
    aspect_att = generate_attention_output_by_attention_weight(text_embed, attend_weight)

    final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                         bias_regularizer=regularizers.l2(0.01))(aspect_att)
    aspect_output = Dense(polarity_class_num, activation="softmax", kernel_regularizer=regularizers.l2(0.01),
                          bias_regularizer=regularizers.l2(0.01))(final_output)
    outputs.append(aspect_output)
    loss.append(losses.binary_crossentropy)


    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    return model


def domain_classifer_attention(max_seq_len, input_dim, output_dim, embedding_matrix, polarity_class_num=3):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding', mask_zero=True)(word_input)

    # aspect category分类
    text_embed = SpatialDropout1D(0.5, name='SpatialDropout1D_embedding')(word_embedding)

    attend_weight = Attention(use_W=True, use_bias=False, u_size=5, W_regularizer=regularizers.l2(0.01),
                              u_regularizer=regularizers.l2(0.01), name='attention')(text_embed)

    aspect_att = generate_attention_output_by_attention_weight(text_embed, attend_weight)

    final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                         bias_regularizer=regularizers.l2(0.01))(aspect_att)
    aspect_output = Dense(polarity_class_num, activation="softmax", kernel_regularizer=regularizers.l2(0.01),
                          bias_regularizer=regularizers.l2(0.01))(final_output)
    outputs.append(aspect_output)
    loss.append(losses.binary_crossentropy)

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    return model


def dc_and_acd(max_seq_len, input_dim, output_dim, embedding_matrix, aspect_class_num=3):
    """at_lstm"""
    inputs = []
    dc_outputs = []
    aspect_outputs = []
    dc_loss = []
    aspect_loss = []

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding', mask_zero=True)(word_input)

    # aspect category分类
    text_embed = SpatialDropout1D(0.5, name='SpatialDropout1D_embedding')(word_embedding)

    hidden_vecs = Bidirectional(LSTM(25, return_sequences=True,
                                     return_state=False,
                                     name='gru',
                                     # kernel_regularizer=regularizers.l2(0.01),
                                     # bias_regularizer=regularizers.l2(0.01)
                                     ))(word_embedding)

    attend_weight = Attention(use_W=True, use_bias=False, u_size=5, W_regularizer=regularizers.l2(0.01),
                              u_regularizer=regularizers.l2(0.01), name='attention')(hidden_vecs)
    aspect_att = generate_attention_output_by_attention_weight(hidden_vecs, attend_weight)

    final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                         bias_regularizer=regularizers.l2(0.01))(aspect_att)
    aspect_output = Dense(2, activation="softmax", kernel_regularizer=regularizers.l2(0.01),
                          bias_regularizer=regularizers.l2(0.01))(final_output)
    dc_outputs.append(aspect_output)
    dc_loss.append(losses.binary_crossentropy)

    acd_input = concatenate([text_embed, hidden_vecs], axis=-1)

    acd_hidden_vecs = Bidirectional(LSTM(25, return_sequences=True,
                                     return_state=False,
                                     name='gru',
                                     # kernel_regularizer=regularizers.l2(0.01),
                                     # bias_regularizer=regularizers.l2(0.01)
                                     ))(acd_input)
    for i in range(aspect_class_num):
        attend_weight = Attention(use_W=True, use_bias=False, u_size=5, W_regularizer=regularizers.l2(0.01),
                                  u_regularizer=regularizers.l2(0.01),
                                  name=('aspect_attention_%d' % i))(acd_hidden_vecs)
        aspect_att = generate_attention_output_by_attention_weight(text_embed, attend_weight)

        final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                             bias_regularizer=regularizers.l2(0.01))(aspect_att)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01),
                              bias_regularizer=regularizers.l2(0.01))(final_output)
        aspect_outputs.append(aspect_output)
        aspect_loss.append(losses.binary_crossentropy)

    model = Model(inputs=inputs, outputs=aspect_outputs + dc_outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=aspect_loss + dc_loss,
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    return model


def slsa_mil(max_sentence_num, max_seq_len, input_dim, output_dim, embedding_matrix, polarity_class_num=3,
             log_attention=False):
    """
    2018-Multiple Instance Learning Networks for Fine-Grained Sentiment Analysis
    :param max_sentence_num:
    :param max_seq_len:
    :param input_dim:
    :param output_dim:
    :param embedding_matrix:
    :param polarity_class_num:
    :return:
    """
    inputs = []
    outputs = []
    loss = []

    # 有多个输入
    inputs = []
    segment_encodings = []
    predicts_based_on_segment =[]
    # 共享的embedding layer
    word_embedding_layer = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                                     name='word_embedding', mask_zero=True)
    for i in range(max_sentence_num):
        input = Input(shape=(max_seq_len,), name=('input_%d' % i))
        inputs.append(input)
        # 对每个输入进行编码
        word_embedding = word_embedding_layer(input)

        # 得到句子编码，这里用attention
        attend_weight = Attention(use_W=True, use_bias=False, u_size=5, W_regularizer=regularizers.l2(0.01),
                                  u_regularizer=regularizers.l2(0.01))(word_embedding)

        if log_attention:
            attend_weight = log_tensor(attend_weight, '\nac_%d:' % 1)
        segment_encoding = generate_attention_output_by_attention_weight(word_embedding, attend_weight)
        segment_encoding = Lambda(lambda x: K.expand_dims(x, axis=1))(segment_encoding)
        segment_encodings.append(segment_encoding)

        # 基于编码预测概率
        probabilities_based_on_segment = Dense(polarity_class_num, name=('predict_%d' % i), activation='softmax',
                                               kernel_regularizer=regularizers.l2(0.01),
                                               bias_regularizer=regularizers.l2(0.01))(segment_encoding)
        # probabilities_based_on_segment = Lambda(lambda x: K.expand_dims(x, axis=1))(probabilities_based_on_segment)
        predicts_based_on_segment.append(probabilities_based_on_segment)

    predicts_based_on_segment_concatenateed = concatenate(predicts_based_on_segment, axis=1)

    segment_encodings_concatenateed = concatenate(segment_encodings, axis=1)
    hidden_vecs = Bidirectional(GRU(100, return_sequences=True,
                                     return_state=False,
                                     name='gru',
                                     # kernel_regularizer=regularizers.l2(0.01),
                                     # bias_regularizer=regularizers.l2(0.01)
                                     ))(segment_encodings_concatenateed)
    attend_weight = Attention(use_W=True, use_bias=False, u_size=5, W_regularizer=regularizers.l2(0.01),
                              u_regularizer=regularizers.l2(0.01))(hidden_vecs)
    output = generate_attention_output_by_attention_weight(predicts_based_on_segment_concatenateed, attend_weight)
    outputs.append(output)

    loss.append(losses.binary_crossentropy)

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    return model


def slsa_mil_bert(max_sentence_num, max_seq_len, input_dim, output_dim, embedding_matrix, bert_dir,
                  polarity_class_num=3, log_attention=False):
    """
    2018-Multiple Instance Learning Networks for Fine-Grained Sentiment Analysis
    :param max_sentence_num:
    :param max_seq_len:
    :param input_dim:
    :param output_dim:
    :param embedding_matrix:
    :param polarity_class_num:
    :return:
    """
    config_path = bert_dir + 'bert_config.json'
    checkpoint_path = bert_dir + 'bert_model.ckpt'
    dict_path = bert_dir + 'vocab.txt'
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=max_seq_len)

    inputs = []
    outputs = []
    loss = []

    # 有多个输入
    inputs = []
    segment_encodings = []
    predicts_based_on_segment =[]
    # 共享的 bert_model
    word_embedding_layer = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                                     name='word_embedding', mask_zero=True)
    for i in range(max_sentence_num):
        input_token = Input(shape=(max_seq_len,), name=('input_token_%d' % i))
        input_segment = Input(shape=(max_seq_len,), name=('input_segment_%d' % i))
        inputs.append(input_token)
        inputs.append(input_segment)
        # 对每个输入进行编码
        segment_encoding = bert_model([input_token, input_segment])

        # 得到句子编码
        segment_encoding = Lambda(lambda x: x[:, 0])(segment_encoding)
        segment_encodings.append(segment_encoding)

        # 基于编码预测概率
        probabilities_based_on_segment = Dense(polarity_class_num, activation='softmax',
                                               kernel_regularizer=regularizers.l2(0.01),
                                               bias_regularizer=regularizers.l2(0.01))(segment_encoding)
        predicts_based_on_segment.append(probabilities_based_on_segment)

    predicts_based_on_segment_concatenateed = concatenate(predicts_based_on_segment, axis=1)

    segment_encodings_concatenateed = concatenate(segment_encodings, axis=1)
    hidden_vecs = Bidirectional(GRU(100, return_sequences=True,
                                     return_state=False,
                                     name='gru',
                                     # kernel_regularizer=regularizers.l2(0.01),
                                     # bias_regularizer=regularizers.l2(0.01)
                                     ))(segment_encodings_concatenateed)
    attend_weight = Attention(use_W=True, use_bias=False, u_size=5, W_regularizer=regularizers.l2(0.01),
                              u_regularizer=regularizers.l2(0.01))(hidden_vecs)
    output = generate_attention_output_by_attention_weight(predicts_based_on_segment_concatenateed, attend_weight)
    outputs.append(output)

    loss.append(losses.binary_crossentropy)

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    return model


def aae_complete_attention(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    aspect_index_and_attention = {}

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding', mask_zero=True)(word_input)

    # aspect category分类
    text_embed = SpatialDropout1D(0.5, name='SpatialDropout1D_embedding')(word_embedding)

    attention_s_dense1 = Dense(300, name='attention_s_dense1')
    for i in range(aspect_class_num):
        aspect_att, sentiment_att = aae_complete_attention_block(text_embed, layer_index=1, aspect_index=i, W_dense=attention_s_dense1)
        if i not in aspect_index_and_attention:
            aspect_index_and_attention[i] = []
        aspect_index_and_attention[i].append((aspect_att, sentiment_att))

    hidden_vecs= Bidirectional(LSTM(100, return_sequences=True,
                                                           return_state=False,
                                                           name='gru',
                                                           # kernel_regularizer=regularizers.l2(0.01),
                                                           # bias_regularizer=regularizers.l2(0.01)
                                                                ))(text_embed)
    hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D_rnn')(hidden_vecs)

    attention_s_dense2 = Dense(200, name='attention_s_dense2')
    for i in range(aspect_class_num):
        aspect_att, sentiment_att = aae_complete_attention_block(hidden_vecs, layer_index=2, aspect_index=i, W_dense=attention_s_dense2)
        if i not in aspect_index_and_attention:
            aspect_index_and_attention[i] = []
        aspect_index_and_attention[i].append((aspect_att, sentiment_att))

    aspect_attentions = []
    for i in range(aspect_class_num):
        # apply attention mechanism
        i_aspect_attentions = [a[0] for a in aspect_index_and_attention[i]]
        if len(i_aspect_attentions) == 1:
            aspect_att_i = i_aspect_attentions[0]
        else:
            aspect_att_i = concatenate(i_aspect_attentions)

        aspect_attentions.append(aspect_att_i)
        final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(aspect_att_i)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(final_output)

        outputs.append(aspect_output)
        loss.append(losses.binary_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[0])

    sentiment_dense = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                            bias_regularizer=regularizers.l2(0.01))
    sentiment_out = Dense(sentiment_class_num, activation='softmax',
                          kernel_regularizer=regularizers.l2(0.1),
                          bias_regularizer=regularizers.l2(0.01))
    for i in range(aspect_class_num):
        i_sentiment_attentions = [a[1] for a in aspect_index_and_attention[i]]
        if len(i_sentiment_attentions) == 1:
            attend_hidden = i_sentiment_attentions[0]
        else:
            attend_hidden = concatenate(i_sentiment_attentions)

        sentiment_dense_output = sentiment_dense(attend_hidden)
        output_layer = sentiment_out(sentiment_dense_output)

        outputs.append(output_layer)
        loss.append(losses.categorical_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[1])

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    return model


def aae_squash(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    aspect_index_and_attention = {}

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding', mask_zero=True)(word_input)
    word_embedding = Lambda(lambda x: capsulelayers.squash_to_one(x))(word_embedding)
    # aspect category分类
    text_embed = SpatialDropout1D(0.5, name='SpatialDropout1D_embedding')(word_embedding)

    for i in range(aspect_class_num):
        aspect_att, sentiment_att = aae_aoa_block(text_embed)
        if i not in aspect_index_and_attention:
            aspect_index_and_attention[i] = []
        aspect_index_and_attention[i].append((aspect_att, sentiment_att))

    hidden_vecs= Bidirectional(LSTM(100, return_sequences=True,
                                                           return_state=False,
                                                           name='gru',
                                                           # kernel_regularizer=regularizers.l2(0.01),
                                                           # bias_regularizer=regularizers.l2(0.01)
                                                                ))(text_embed)
    hidden_vecs = Lambda(lambda x: capsulelayers.squash_to_one(x))(hidden_vecs)
    hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D_rnn')(hidden_vecs)

    for i in range(aspect_class_num):
        aspect_att, sentiment_att = aae_aoa_block(hidden_vecs)
        if i not in aspect_index_and_attention:
            aspect_index_and_attention[i] = []
        aspect_index_and_attention[i].append((aspect_att, sentiment_att))

    aspect_attentions = []
    for i in range(aspect_class_num):
        # apply attention mechanism
        i_aspect_attentions = [a[0] for a in aspect_index_and_attention[i]]
        if len(i_aspect_attentions) == 1:
            aspect_att_i = i_aspect_attentions[0]
        else:
            aspect_att_i = concatenate(i_aspect_attentions)

        aspect_attentions.append(aspect_att_i)
        final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(aspect_att_i)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(final_output)

        outputs.append(aspect_output)
        loss.append(losses.binary_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[0])

    sentiment_dense = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                            bias_regularizer=regularizers.l2(0.01))
    sentiment_out = Dense(sentiment_class_num, activation='softmax',
                          kernel_regularizer=regularizers.l2(0.1),
                          bias_regularizer=regularizers.l2(0.01))
    for i in range(aspect_class_num):
        i_sentiment_attentions = [a[1] for a in aspect_index_and_attention[i]]
        if len(i_sentiment_attentions) == 1:
            attend_hidden = i_sentiment_attentions[0]
        else:
            attend_hidden = concatenate(i_sentiment_attentions)

        sentiment_dense_output = sentiment_dense(attend_hidden)
        output_layer = sentiment_out(sentiment_dense_output)

        outputs.append(output_layer)
        loss.append(losses.categorical_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[1])

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    return model


def aae_share(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    aspect_index_and_attention = {}

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding', mask_zero=True)(word_input)

    # aspect category分类
    text_embed = SpatialDropout1D(0.5, name='SpatialDropout1D_embedding')(word_embedding)

    for i in range(aspect_class_num):
        aspect_att, sentiment_att = aae_aoa_block(text_embed)
        if i not in aspect_index_and_attention:
            aspect_index_and_attention[i] = []
        aspect_index_and_attention[i].append((aspect_att, sentiment_att))

    hidden_vecs= Bidirectional(LSTM(100, return_sequences=True,
                                                           return_state=False,
                                                           name='gru',
                                                           # kernel_regularizer=regularizers.l2(0.01),
                                                           # bias_regularizer=regularizers.l2(0.01)
                                                                ))(text_embed)
    hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D_rnn')(hidden_vecs)

    for i in range(aspect_class_num):
        aspect_att, sentiment_att = aae_aoa_block(hidden_vecs)
        if i not in aspect_index_and_attention:
            aspect_index_and_attention[i] = []
        aspect_index_and_attention[i].append((aspect_att, sentiment_att))

    aspect_attentions = []
    for i in range(aspect_class_num):
        # apply attention mechanism
        i_aspect_attentions = [a[0] for a in aspect_index_and_attention[i]]
        if len(i_aspect_attentions) == 1:
            aspect_att_i = i_aspect_attentions[0]
        else:
            aspect_att_i = concatenate(i_aspect_attentions)

        aspect_attentions.append(aspect_att_i)
        final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(aspect_att_i)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(final_output)

        outputs.append(aspect_output)
        loss.append(losses.binary_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[0])

    sentiment_dense = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                            bias_regularizer=regularizers.l2(0.01))
    sentiment_out = Dense(sentiment_class_num, activation='softmax',
                          kernel_regularizer=regularizers.l2(0.1),
                          bias_regularizer=regularizers.l2(0.01))
    for i in range(aspect_class_num):
        i_sentiment_attentions = [a[0] for a in aspect_index_and_attention[i]]
        if len(i_sentiment_attentions) == 1:
            attend_hidden = i_sentiment_attentions[0]
        else:
            attend_hidden = concatenate(i_sentiment_attentions)

        sentiment_dense_output = sentiment_dense(attend_hidden)
        output_layer = sentiment_out(sentiment_dense_output)

        outputs.append(output_layer)
        loss.append(losses.categorical_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[1])

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    return model


def aae_at_lstm(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    aspect_index_and_attention = {}

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding', mask_zero=True)(word_input)

    # aspect category分类
    text_embed = SpatialDropout1D(0.5, name='SpatialDropout1D_embedding')(word_embedding)

    share_attenion_layer1 = Attention(use_W=True, use_bias=False, u_size=50, W_regularizer=regularizers.l2(0.01),
                              u_regularizer=regularizers.l2(0.01))
    for i in range(aspect_class_num):
        aspect_att, sentiment_att = aae_at_lstm_block(text_embed, max_seq_len, share_attenion_layer1)
        if i not in aspect_index_and_attention:
            aspect_index_and_attention[i] = []
        aspect_index_and_attention[i].append((aspect_att, sentiment_att))

    hidden_vecs= Bidirectional(LSTM(100, return_sequences=True,
                                                           return_state=False,
                                                           name='gru',
                                                           # kernel_regularizer=regularizers.l2(0.01),
                                                           # bias_regularizer=regularizers.l2(0.01)
                                                                ))(text_embed)
    hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D_rnn')(hidden_vecs)

    share_attenion_layer2 = Attention(use_W=True, use_bias=False, u_size=50, W_regularizer=regularizers.l2(0.01),
                              u_regularizer=regularizers.l2(0.01))
    for i in range(aspect_class_num):
        aspect_att, sentiment_att = aae_at_lstm_block(hidden_vecs, max_seq_len, share_attenion_layer2)
        if i not in aspect_index_and_attention:
            aspect_index_and_attention[i] = []
        aspect_index_and_attention[i].append((aspect_att, sentiment_att))

    aspect_attentions = []
    for i in range(aspect_class_num):
        # apply attention mechanism
        i_aspect_attentions = [a[0] for a in aspect_index_and_attention[i]]
        if len(i_aspect_attentions) == 1:
            aspect_att_i = i_aspect_attentions[0]
        else:
            aspect_att_i = concatenate(i_aspect_attentions)

        aspect_attentions.append(aspect_att_i)
        final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(aspect_att_i)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(final_output)

        outputs.append(aspect_output)
        loss.append(losses.binary_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[0])

    sentiment_dense = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                            bias_regularizer=regularizers.l2(0.01))
    sentiment_out = Dense(sentiment_class_num, activation='softmax',
                          kernel_regularizer=regularizers.l2(0.1),
                          bias_regularizer=regularizers.l2(0.01))
    for i in range(aspect_class_num):
        i_sentiment_attentions = [a[1] for a in aspect_index_and_attention[i]]
        if len(i_sentiment_attentions) == 1:
            attend_hidden = i_sentiment_attentions[0]
        else:
            attend_hidden = concatenate(i_sentiment_attentions)

        sentiment_dense_output = sentiment_dense(attend_hidden)
        output_layer = sentiment_out(sentiment_dense_output)

        outputs.append(output_layer)
        loss.append(losses.categorical_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[1])

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    return model


def aae_interactive(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    aspect_index_and_attention = {}

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding', mask_zero=True)(word_input)

    # aspect category分类
    text_embed = SpatialDropout1D(0.5, name='SpatialDropout1D_embedding')(word_embedding)

    for i in range(aspect_class_num):
        aspect_att, sentiment_att = aae_aoa_block(text_embed)
        if i not in aspect_index_and_attention:
            aspect_index_and_attention[i] = []
        aspect_index_and_attention[i].append((aspect_att, sentiment_att))

    hidden_vecs= Bidirectional(LSTM(100, return_sequences=True,
                                                           return_state=False,
                                                           name='gru',
                                                           # kernel_regularizer=regularizers.l2(0.01),
                                                           # bias_regularizer=regularizers.l2(0.01)
                                                                ))(text_embed)
    hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D_rnn')(hidden_vecs)

    for i in range(aspect_class_num):
        aspect_att, sentiment_att = aae_aoa_block(hidden_vecs)
        if i not in aspect_index_and_attention:
            aspect_index_and_attention[i] = []
        aspect_index_and_attention[i].append((aspect_att, sentiment_att))

    aspect_attentions = []
    for i in range(aspect_class_num):
        # apply attention mechanism
        i_aspect_attentions = [a[0] for a in aspect_index_and_attention[i]]
        if len(i_aspect_attentions) == 1:
            aspect_att_i = i_aspect_attentions[0]
        else:
            aspect_att_i = concatenate(i_aspect_attentions)

        aspect_attentions.append(aspect_att_i)
        final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(aspect_att_i)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(final_output)

        outputs.append(aspect_output)
        loss.append(losses.binary_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[0])

    sentiment_dense = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                            bias_regularizer=regularizers.l2(0.01))
    sentiment_out = Dense(sentiment_class_num, activation='softmax',
                          kernel_regularizer=regularizers.l2(0.1),
                          bias_regularizer=regularizers.l2(0.01))
    for i in range(aspect_class_num):
        i_sentiment_attentions = [Add()([a[0], a[1]]) for a in aspect_index_and_attention[i]]
        if len(i_sentiment_attentions) == 1:
            attend_hidden = i_sentiment_attentions[0]
        else:
            attend_hidden = concatenate(i_sentiment_attentions)

        sentiment_dense_output = sentiment_dense(attend_hidden)
        output_layer = sentiment_out(sentiment_dense_output)

        outputs.append(output_layer)
        loss.append(losses.categorical_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[1])

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    return model


def aae_interactive_concat(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    aspect_index_and_attention = {}

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding', mask_zero=True)(word_input)

    # aspect category分类
    text_embed = SpatialDropout1D(0.5, name='SpatialDropout1D_embedding')(word_embedding)

    for i in range(aspect_class_num):
        aspect_att, sentiment_att = aae_aoa_block(text_embed)
        if i not in aspect_index_and_attention:
            aspect_index_and_attention[i] = []
        aspect_index_and_attention[i].append((aspect_att, sentiment_att))

    hidden_vecs= Bidirectional(LSTM(100, return_sequences=True,
                                                           return_state=False,
                                                           name='gru',
                                                           # kernel_regularizer=regularizers.l2(0.01),
                                                           # bias_regularizer=regularizers.l2(0.01)
                                                                ))(text_embed)
    hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D_rnn')(hidden_vecs)

    for i in range(aspect_class_num):
        aspect_att, sentiment_att = aae_aoa_block(hidden_vecs)
        if i not in aspect_index_and_attention:
            aspect_index_and_attention[i] = []
        aspect_index_and_attention[i].append((aspect_att, sentiment_att))

    aspect_attentions = []
    for i in range(aspect_class_num):
        # apply attention mechanism
        i_aspect_attentions = [a[0] for a in aspect_index_and_attention[i]]
        if len(i_aspect_attentions) == 1:
            aspect_att_i = i_aspect_attentions[0]
        else:
            aspect_att_i = concatenate(i_aspect_attentions)

        aspect_attentions.append(aspect_att_i)
        final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(aspect_att_i)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(final_output)

        outputs.append(aspect_output)
        loss.append(losses.binary_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[0])

    sentiment_dense = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                            bias_regularizer=regularizers.l2(0.01))
    sentiment_out = Dense(sentiment_class_num, activation='softmax',
                          kernel_regularizer=regularizers.l2(0.1),
                          bias_regularizer=regularizers.l2(0.01))
    for i in range(aspect_class_num):
        i_sentiment_attentions = [Concatenate()([a[0], a[1]]) for a in aspect_index_and_attention[i]]
        if len(i_sentiment_attentions) == 1:
            attend_hidden = i_sentiment_attentions[0]
        else:
            attend_hidden = concatenate(i_sentiment_attentions)

        sentiment_dense_output = sentiment_dense(attend_hidden)
        output_layer = sentiment_out(sentiment_dense_output)

        outputs.append(output_layer)
        loss.append(losses.categorical_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[1])

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    return model


def aae_single(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    aspect_index_and_attention = {}

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding', mask_zero=True)(word_input)

    # aspect category分类
    text_embed = SpatialDropout1D(0.5, name='SpatialDropout1D_embedding')(word_embedding)

    # for i in range(aspect_class_num):
    #     aspect_att, sentiment_att = aoa_block(text_embed)
    #     if i not in aspect_index_and_attention:
    #         aspect_index_and_attention[i] = []
    #     aspect_index_and_attention[i].append((aspect_att, sentiment_att))

    hidden_vecs= Bidirectional(LSTM(100, return_sequences=True,
                                                           return_state=False,
                                                           name='gru',
                                                           # kernel_regularizer=regularizers.l2(0.01),
                                                           # bias_regularizer=regularizers.l2(0.01)
                                                                ))(text_embed)
    hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D_rnn')(hidden_vecs)

    for i in range(aspect_class_num):
        aspect_att, sentiment_att = aae_aoa_block(hidden_vecs)
        if i not in aspect_index_and_attention:
            aspect_index_and_attention[i] = []
        aspect_index_and_attention[i].append((aspect_att, sentiment_att))

    aspect_attentions = []
    for i in range(aspect_class_num):
        # apply attention mechanism
        i_aspect_attentions = [a[0] for a in aspect_index_and_attention[i]]
        if len(i_aspect_attentions) == 1:
            aspect_att_i = i_aspect_attentions[0]
        else:
            aspect_att_i = concatenate(i_aspect_attentions)

        aspect_attentions.append(aspect_att_i)
        final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(aspect_att_i)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(final_output)

        outputs.append(aspect_output)
        loss.append(losses.binary_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[0])

    sentiment_dense = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                            bias_regularizer=regularizers.l2(0.01))
    sentiment_out = Dense(sentiment_class_num, activation='softmax',
                          kernel_regularizer=regularizers.l2(0.1),
                          bias_regularizer=regularizers.l2(0.01))
    for i in range(aspect_class_num):
        i_sentiment_attentions = [a[1] for a in aspect_index_and_attention[i]]
        if len(i_sentiment_attentions) == 1:
            attend_hidden = i_sentiment_attentions[0]
        else:
            attend_hidden = concatenate(i_sentiment_attentions)

        sentiment_dense_output = sentiment_dense(attend_hidden)
        output_layer = sentiment_out(sentiment_dense_output)

        outputs.append(output_layer)
        loss.append(losses.categorical_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[1])

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    return model


def aae_without_aae(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    aspect_index_and_attention = {}

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding', mask_zero=True)(word_input)
    # word_embedding = Embedding(input_dim, output_dim, trainable=False,
    #                            name='word_embedding1', mask_zero=True)(word_input)

    # aspect category分类
    text_embed = SpatialDropout1D(0.5, name='SpatialDropout1D_embedding')(word_embedding)

    for i in range(aspect_class_num):
        input_aspect = Input(shape=(1,), )
        inputs.append(input_aspect)

    import keras
    keras.initializers.uniform
    asp_embedding1 = Embedding(input_dim=aspect_class_num + 1, output_dim=300, name='asp_embedding1',
                               trainable=True)
    for i in range(aspect_class_num):
        aspect_embed = asp_embedding1(inputs[i + 1])
        # reshape to 2d
        aspect_embed = Flatten()(aspect_embed)
        # aspect_embed = log_tensor(aspect_embed, 'aspect_embed')

        aspect_att, sentiment_att = aoa_block_with_fixed_ae(text_embed, aspect_embed)
        if i not in aspect_index_and_attention:
            aspect_index_and_attention[i] = []
        aspect_index_and_attention[i].append((aspect_att, sentiment_att))

    hidden_vecs= Bidirectional(LSTM(100, return_sequences=True,
                                                           return_state=False,
                                                           name='gru',
                                                           # kernel_regularizer=regularizers.l2(0.01),
                                                           # bias_regularizer=regularizers.l2(0.01)
                                                                ))(text_embed)
    hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D_rnn')(hidden_vecs)

    asp_embedding2 = Embedding(input_dim=aspect_class_num + 1, output_dim=200, name='asp_embedding2',
                               trainable=True)
    for i in range(aspect_class_num):
        aspect_embed = asp_embedding2(inputs[i + 1])
        # reshape to 2d
        aspect_embed = Flatten()(aspect_embed)

        aspect_att, sentiment_att = aoa_block_with_fixed_ae(hidden_vecs, aspect_embed)
        # sentiment_att = log_tensor(aspect_att, 'sentiment_att')
        if i not in aspect_index_and_attention:
            aspect_index_and_attention[i] = []
        aspect_index_and_attention[i].append((aspect_att, sentiment_att))

    aspect_attentions = []
    for i in range(aspect_class_num):
        # apply attention mechanism
        i_aspect_attentions = [a[0] for a in aspect_index_and_attention[i]]
        if len(i_aspect_attentions) == 1:
            aspect_att_i = i_aspect_attentions[0]
        else:
            aspect_att_i = concatenate(i_aspect_attentions)

        aspect_attentions.append(aspect_att_i)
        final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(aspect_att_i)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(final_output)

        outputs.append(aspect_output)
        loss.append(losses.binary_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[0])

    sentiment_dense = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                            bias_regularizer=regularizers.l2(0.01))
    sentiment_out = Dense(sentiment_class_num, activation='softmax',
                          kernel_regularizer=regularizers.l2(0.1),
                          bias_regularizer=regularizers.l2(0.01))
    for i in range(aspect_class_num):
        i_sentiment_attentions = [a[1] for a in aspect_index_and_attention[i]]
        if len(i_sentiment_attentions) == 1:
            attend_hidden = i_sentiment_attentions[0]
        else:
            attend_hidden = concatenate(i_sentiment_attentions)

        sentiment_dense_output = sentiment_dense(attend_hidden)
        output_layer = sentiment_out(sentiment_dense_output)

        outputs.append(output_layer)
        loss.append(losses.categorical_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[1])

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    # 解决调用 GraphViz 失败
    # https://blog.csdn.net/sinat_40282753/article/details/85046871
    # plot_model(model, to_file=data_path.data_base_dir + task_conf.current_dataset + '_aae_without_aae.png')
    return model


def aae_without_aae_share(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    aspect_index_and_attention = {}

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding', mask_zero=True)(word_input)
    # word_embedding = Embedding(input_dim, output_dim, trainable=False,
    #                            name='word_embedding1', mask_zero=True)(word_input)

    # aspect category分类
    text_embed = SpatialDropout1D(0.5, name='SpatialDropout1D_embedding')(word_embedding)

    for i in range(aspect_class_num):
        input_aspect = Input(shape=(1,), )
        inputs.append(input_aspect)

    asp_embedding1 = Embedding(input_dim=aspect_class_num + 1, output_dim=300, name='asp_embedding1',
                               trainable=True)
    for i in range(aspect_class_num):
        aspect_embed = asp_embedding1(inputs[i + 1])
        # reshape to 2d
        aspect_embed = Flatten()(aspect_embed)
        # aspect_embed = log_tensor(aspect_embed, 'aspect_embed')

        aspect_att, sentiment_att = aoa_block_with_fixed_ae(text_embed, aspect_embed)
        if i not in aspect_index_and_attention:
            aspect_index_and_attention[i] = []
        aspect_index_and_attention[i].append((aspect_att, sentiment_att))

    hidden_vecs= Bidirectional(LSTM(100, return_sequences=True,
                                                           return_state=False,
                                                           name='gru',
                                                           # kernel_regularizer=regularizers.l2(0.01),
                                                           # bias_regularizer=regularizers.l2(0.01)
                                                                ))(text_embed)
    hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D_rnn')(hidden_vecs)

    asp_embedding2 = Embedding(input_dim=aspect_class_num + 1, output_dim=200, name='asp_embedding2',
                               trainable=True)
    for i in range(aspect_class_num):
        aspect_embed = asp_embedding2(inputs[i + 1])
        # reshape to 2d
        aspect_embed = Flatten()(aspect_embed)

        aspect_att, sentiment_att = aoa_block_with_fixed_ae(hidden_vecs, aspect_embed)
        # sentiment_att = log_tensor(aspect_att, 'sentiment_att')
        if i not in aspect_index_and_attention:
            aspect_index_and_attention[i] = []
        aspect_index_and_attention[i].append((aspect_att, sentiment_att))

    aspect_attentions = []
    for i in range(aspect_class_num):
        # apply attention mechanism
        i_aspect_attentions = [a[0] for a in aspect_index_and_attention[i]]
        if len(i_aspect_attentions) == 1:
            aspect_att_i = i_aspect_attentions[0]
        else:
            aspect_att_i = concatenate(i_aspect_attentions)

        aspect_attentions.append(aspect_att_i)
        final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(aspect_att_i)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(final_output)

        outputs.append(aspect_output)
        loss.append(losses.binary_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[0])

    for i in range(aspect_class_num):
        i_sentiment_attentions = [a[1] for a in aspect_index_and_attention[i]]
        if len(i_sentiment_attentions) == 1:
            attend_hidden = i_sentiment_attentions[0]
        else:
            attend_hidden = concatenate(i_sentiment_attentions)

        sentiment_dense_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                            bias_regularizer=regularizers.l2(0.01))(attend_hidden)
        output_layer = Dense(sentiment_class_num, activation='softmax',
                          kernel_regularizer=regularizers.l2(0.1),
                          bias_regularizer=regularizers.l2(0.01))(sentiment_dense_output)

        outputs.append(output_layer)
        loss.append(losses.categorical_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[1])

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    # 解决调用 GraphViz 失败
    # https://blog.csdn.net/sinat_40282753/article/details/85046871
    # plot_model(model, to_file=data_path.data_base_dir + task_conf.current_dataset + '_aae_without_aae.png')
    return model


def acaoa_semeval_2016_task_5_lapt_sb2_ada_sh_no_com(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    aspect_index_and_attention = {}

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding', mask_zero=True)(word_input)

    # aspect category分类
    text_embed = SpatialDropout1D(0.5, name='SpatialDropout1D_embedding')(word_embedding)

    for i in range(aspect_class_num):
        aspect_att, sentiment_att = aoa_block_no_com(text_embed)
        if i not in aspect_index_and_attention:
            aspect_index_and_attention[i] = []
        aspect_index_and_attention[i].append((aspect_att, sentiment_att))

    hidden_vecs= Bidirectional(LSTM(100, return_sequences=True,
                                                           return_state=False,
                                                           name='gru',
                                                           # kernel_regularizer=regularizers.l2(0.01),
                                                           # bias_regularizer=regularizers.l2(0.01)
                                                                ))(text_embed)
    hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D_rnn')(hidden_vecs)

    for i in range(aspect_class_num):
        aspect_att, sentiment_att = aoa_block_no_com(hidden_vecs)
        if i not in aspect_index_and_attention:
            aspect_index_and_attention[i] = []
        aspect_index_and_attention[i].append((aspect_att, sentiment_att))

    aspect_attentions = []
    for i in range(aspect_class_num):
        # apply attention mechanism
        i_aspect_attentions = [a[0] for a in aspect_index_and_attention[i]]
        if len(i_aspect_attentions) == 1:
            aspect_att_i = i_aspect_attentions[0]
        else:
            aspect_att_i = concatenate(i_aspect_attentions)

        aspect_attentions.append(aspect_att_i)
        final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(aspect_att_i)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(final_output)

        outputs.append(aspect_output)
        loss.append(losses.binary_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[0])

    sentiment_dense = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                            bias_regularizer=regularizers.l2(0.01))
    sentiment_out = Dense(sentiment_class_num, activation='softmax',
                          kernel_regularizer=regularizers.l2(0.1),
                          bias_regularizer=regularizers.l2(0.01))
    for i in range(aspect_class_num):
        i_sentiment_attentions = [a[1] for a in aspect_index_and_attention[i]]
        if len(i_sentiment_attentions) == 1:
            attend_hidden = i_aspect_attentions[0]
        else:
            attend_hidden = concatenate(i_aspect_attentions)

        sentiment_dense_output = sentiment_dense(attend_hidden)
        output_layer = sentiment_out(sentiment_dense_output)

        outputs.append(output_layer)
        loss.append(losses.categorical_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[1])

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file=data_path.data_base_dir + 'semeval_2014_task_4_rest_acaoa.png')
    return model


def aae_ada_sh_no_com(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    aspect_index_and_attention = {}

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding', mask_zero=True)(word_input)

    # aspect category分类
    text_embed = SpatialDropout1D(0.5, name='SpatialDropout1D_embedding')(word_embedding)

    for i in range(aspect_class_num):
        aspect_att, sentiment_att = aoa_block_no_com(text_embed)
        if i not in aspect_index_and_attention:
            aspect_index_and_attention[i] = []
        aspect_index_and_attention[i].append((aspect_att, sentiment_att))

    hidden_vecs= Bidirectional(LSTM(100, return_sequences=True,
                                                           return_state=False,
                                                           name='gru',
                                                           # kernel_regularizer=regularizers.l2(0.01),
                                                           # bias_regularizer=regularizers.l2(0.01)
                                                                ))(text_embed)
    hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D_rnn')(hidden_vecs)

    for i in range(aspect_class_num):
        aspect_att, sentiment_att = aoa_block_no_com(hidden_vecs)
        if i not in aspect_index_and_attention:
            aspect_index_and_attention[i] = []
        aspect_index_and_attention[i].append((aspect_att, sentiment_att))

    aspect_attentions = []
    for i in range(aspect_class_num):
        # apply attention mechanism
        i_aspect_attentions = [a[0] for a in aspect_index_and_attention[i]]
        if len(i_aspect_attentions) == 1:
            aspect_att_i = i_aspect_attentions[0]
        else:
            aspect_att_i = concatenate(i_aspect_attentions)

        aspect_attentions.append(aspect_att_i)
        final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(aspect_att_i)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(final_output)

        outputs.append(aspect_output)
        loss.append(losses.binary_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[0])

    sentiment_dense = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                            bias_regularizer=regularizers.l2(0.01))
    sentiment_out = Dense(sentiment_class_num, activation='softmax',
                          kernel_regularizer=regularizers.l2(0.1),
                          bias_regularizer=regularizers.l2(0.01))
    for i in range(aspect_class_num):
        i_sentiment_attentions = [a[1] for a in aspect_index_and_attention[i]]
        if len(i_sentiment_attentions) == 1:
            attend_hidden = i_sentiment_attentions[0]
        else:
            attend_hidden = concatenate(i_sentiment_attentions)

        sentiment_dense_output = sentiment_dense(attend_hidden)
        output_layer = sentiment_out(sentiment_dense_output)

        outputs.append(output_layer)
        loss.append(losses.categorical_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[1])

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file=data_path.data_base_dir + 'semeval_2014_task_4_rest_acaoa.png')
    return model


def acaoa_semeval_2016_task_5_lapt_sb2_ada_no_sh_no_com(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    aspect_index_and_attention = {}

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding', mask_zero=True)(word_input)

    # aspect category分类
    text_embed = SpatialDropout1D(0.5, name='SpatialDropout1D_embedding')(word_embedding)

    for i in range(aspect_class_num):
        aspect_att, sentiment_att = aoa_block_no_com(text_embed)
        if i not in aspect_index_and_attention:
            aspect_index_and_attention[i] = []
        aspect_index_and_attention[i].append((aspect_att, sentiment_att))

    hidden_vecs= Bidirectional(LSTM(100, return_sequences=True,
                                                           return_state=False,
                                                           name='gru',
                                                           # kernel_regularizer=regularizers.l2(0.01),
                                                           # bias_regularizer=regularizers.l2(0.01)
                                                                ))(text_embed)
    hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D_rnn')(hidden_vecs)

    for i in range(aspect_class_num):
        aspect_att, sentiment_att = aoa_block_no_com(hidden_vecs)
        if i not in aspect_index_and_attention:
            aspect_index_and_attention[i] = []
        aspect_index_and_attention[i].append((aspect_att, sentiment_att))

    aspect_attentions = []
    for i in range(aspect_class_num):
        # apply attention mechanism
        i_aspect_attentions = [a[0] for a in aspect_index_and_attention[i]]
        if len(i_aspect_attentions) == 1:
            aspect_att_i = i_aspect_attentions[0]
        else:
            aspect_att_i = concatenate(i_aspect_attentions)

        aspect_attentions.append(aspect_att_i)
        final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(aspect_att_i)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(final_output)

        outputs.append(aspect_output)
        loss.append(losses.binary_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[0])

    for i in range(aspect_class_num):
        i_sentiment_attentions = [a[1] for a in aspect_index_and_attention[i]]
        if len(i_sentiment_attentions) == 1:
            attend_hidden = i_aspect_attentions[0]
        else:
            attend_hidden = concatenate(i_aspect_attentions)

        sentiment_dense_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                            bias_regularizer=regularizers.l2(0.01))(attend_hidden)
        output_layer = Dense(sentiment_class_num, activation='softmax',
                          kernel_regularizer=regularizers.l2(0.1),
                          bias_regularizer=regularizers.l2(0.01))(sentiment_dense_output)

        outputs.append(output_layer)
        loss.append(losses.categorical_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[1])

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file=data_path.data_base_dir + 'semeval_2014_task_4_rest_acaoa.png')
    return model


def aae_ada_no_sh_no_com(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    aspect_index_and_attention = {}

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding', mask_zero=True)(word_input)

    # aspect category分类
    text_embed = SpatialDropout1D(0.5, name='SpatialDropout1D_embedding')(word_embedding)

    for i in range(aspect_class_num):
        aspect_att, sentiment_att = aoa_block_no_com(text_embed)
        if i not in aspect_index_and_attention:
            aspect_index_and_attention[i] = []
        aspect_index_and_attention[i].append((aspect_att, sentiment_att))

    hidden_vecs= Bidirectional(LSTM(100, return_sequences=True,
                                                           return_state=False,
                                                           name='gru',
                                                           # kernel_regularizer=regularizers.l2(0.01),
                                                           # bias_regularizer=regularizers.l2(0.01)
                                                                ))(text_embed)
    hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D_rnn')(hidden_vecs)

    for i in range(aspect_class_num):
        aspect_att, sentiment_att = aoa_block_no_com(hidden_vecs)
        if i not in aspect_index_and_attention:
            aspect_index_and_attention[i] = []
        aspect_index_and_attention[i].append((aspect_att, sentiment_att))

    aspect_attentions = []
    for i in range(aspect_class_num):
        # apply attention mechanism
        i_aspect_attentions = [a[0] for a in aspect_index_and_attention[i]]
        if len(i_aspect_attentions) == 1:
            aspect_att_i = i_aspect_attentions[0]
        else:
            aspect_att_i = concatenate(i_aspect_attentions)

        aspect_attentions.append(aspect_att_i)
        final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(aspect_att_i)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(final_output)

        outputs.append(aspect_output)
        loss.append(losses.binary_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[0])

    for i in range(aspect_class_num):
        i_sentiment_attentions = [a[1] for a in aspect_index_and_attention[i]]
        if len(i_sentiment_attentions) == 1:
            attend_hidden = i_sentiment_attentions[0]
        else:
            attend_hidden = concatenate(i_sentiment_attentions)

        sentiment_dense_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                            bias_regularizer=regularizers.l2(0.01))(attend_hidden)
        output_layer = Dense(sentiment_class_num, activation='softmax',
                          kernel_regularizer=regularizers.l2(0.1),
                          bias_regularizer=regularizers.l2(0.01))(sentiment_dense_output)

        outputs.append(output_layer)
        loss.append(losses.categorical_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[1])

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file=data_path.data_base_dir + 'semeval_2014_task_4_rest_acaoa.png')
    return model


def acaoa_semeval_2016_task_5_lapt_sb2_s_ada_no_sh_no_com(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    aspect_index_and_attention = {}

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding', mask_zero=True)(word_input)

    # aspect category分类
    text_embed = SpatialDropout1D(0.5, name='SpatialDropout1D_embedding')(word_embedding)

    # for i in range(aspect_class_num):
    #     aspect_att, sentiment_att = aoa_block_no_com(text_embed)
    #     if i not in aspect_index_and_attention:
    #         aspect_index_and_attention[i] = []
    #     aspect_index_and_attention[i].append((aspect_att, sentiment_att))

    hidden_vecs= Bidirectional(LSTM(100, return_sequences=True,
                                                           return_state=False,
                                                           name='gru',
                                                           # kernel_regularizer=regularizers.l2(0.01),
                                                           # bias_regularizer=regularizers.l2(0.01)
                                                                ))(text_embed)
    hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D_rnn')(hidden_vecs)

    for i in range(aspect_class_num):
        aspect_att, sentiment_att = aoa_block_no_com(hidden_vecs)
        if i not in aspect_index_and_attention:
            aspect_index_and_attention[i] = []
        aspect_index_and_attention[i].append((aspect_att, sentiment_att))

    aspect_attentions = []
    for i in range(aspect_class_num):
        # apply attention mechanism
        i_aspect_attentions = [a[0] for a in aspect_index_and_attention[i]]
        if len(i_aspect_attentions) == 1:
            aspect_att_i = i_aspect_attentions[0]
        else:
            aspect_att_i = concatenate(i_aspect_attentions)

        aspect_attentions.append(aspect_att_i)
        final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(aspect_att_i)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(final_output)

        outputs.append(aspect_output)
        loss.append(losses.binary_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[0])

    for i in range(aspect_class_num):
        i_sentiment_attentions = [a[1] for a in aspect_index_and_attention[i]]
        if len(i_sentiment_attentions) == 1:
            attend_hidden = i_aspect_attentions[0]
        else:
            attend_hidden = concatenate(i_aspect_attentions)

        sentiment_dense_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                            bias_regularizer=regularizers.l2(0.01))(attend_hidden)
        output_layer = Dense(sentiment_class_num, activation='softmax',
                          kernel_regularizer=regularizers.l2(0.1),
                          bias_regularizer=regularizers.l2(0.01))(sentiment_dense_output)

        outputs.append(output_layer)
        loss.append(losses.categorical_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[1])

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file=data_path.data_base_dir + 'semeval_2014_task_4_rest_acaoa.png')
    return model


def aae_s_ada_no_sh_no_com(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    aspect_index_and_attention = {}

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding', mask_zero=True)(word_input)

    # aspect category分类
    text_embed = SpatialDropout1D(0.5, name='SpatialDropout1D_embedding')(word_embedding)

    # for i in range(aspect_class_num):
    #     aspect_att, sentiment_att = aoa_block_no_com(text_embed)
    #     if i not in aspect_index_and_attention:
    #         aspect_index_and_attention[i] = []
    #     aspect_index_and_attention[i].append((aspect_att, sentiment_att))

    hidden_vecs= Bidirectional(LSTM(100, return_sequences=True,
                                                           return_state=False,
                                                           name='gru',
                                                           # kernel_regularizer=regularizers.l2(0.01),
                                                           # bias_regularizer=regularizers.l2(0.01)
                                                                ))(text_embed)
    hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D_rnn')(hidden_vecs)

    for i in range(aspect_class_num):
        aspect_att, sentiment_att = aoa_block_no_com(hidden_vecs)
        if i not in aspect_index_and_attention:
            aspect_index_and_attention[i] = []
        aspect_index_and_attention[i].append((aspect_att, sentiment_att))

    aspect_attentions = []
    for i in range(aspect_class_num):
        # apply attention mechanism
        i_aspect_attentions = [a[0] for a in aspect_index_and_attention[i]]
        if len(i_aspect_attentions) == 1:
            aspect_att_i = i_aspect_attentions[0]
        else:
            aspect_att_i = concatenate(i_aspect_attentions)

        aspect_attentions.append(aspect_att_i)
        final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(aspect_att_i)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(final_output)

        outputs.append(aspect_output)
        loss.append(losses.binary_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[0])

    for i in range(aspect_class_num):
        i_sentiment_attentions = [a[1] for a in aspect_index_and_attention[i]]
        if len(i_sentiment_attentions) == 1:
            attend_hidden = i_sentiment_attentions[0]
        else:
            attend_hidden = concatenate(i_sentiment_attentions)

        sentiment_dense_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                            bias_regularizer=regularizers.l2(0.01))(attend_hidden)
        output_layer = Dense(sentiment_class_num, activation='softmax',
                          kernel_regularizer=regularizers.l2(0.1),
                          bias_regularizer=regularizers.l2(0.01))(sentiment_dense_output)

        outputs.append(output_layer)
        loss.append(losses.categorical_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[1])

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file=data_path.data_base_dir + 'semeval_2014_task_4_rest_acaoa.png')
    return model


def acaoa_semeval_2016_task_5_lapt_sb2_s_no_ada_no_sh_no_com(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    aspect_index_and_attention = {}

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding', mask_zero=True)(word_input)

    # aspect category分类
    text_embed = SpatialDropout1D(0.5, name='SpatialDropout1D_embedding')(word_embedding)

    hidden_vecs= Bidirectional(LSTM(100, return_sequences=True,
                                                           return_state=False,
                                                           name='gru',
                                                           # kernel_regularizer=regularizers.l2(0.01),
                                                           # bias_regularizer=regularizers.l2(0.01)
                                                                ))(text_embed)
    hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D_rnn')(hidden_vecs)

    for i in range(aspect_class_num):
        # apply attention mechanism
        attend_weight = Attention(use_W=False, W_regularizer=regularizers.l2(0.01),
                                  u_regularizer=regularizers.l2(0.01))(hidden_vecs)
        aspect_att_i = generate_attention_output_by_attention_weight(hidden_vecs, attend_weight)

        final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(aspect_att_i)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(final_output)
        outputs.append(aspect_output)
        loss.append(losses.binary_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[0])

    asp_embedding = Embedding(input_dim=aspect_class_num + 1, output_dim=200, trainable=True)
    for i in range(aspect_class_num):
        input_aspect = Input(shape=(1,), )
        inputs.append(input_aspect)
        aspect_embed = asp_embedding(input_aspect)
        # reshape to 2d
        aspect_embed = Flatten()(aspect_embed)
        # repeat aspect for every word in sequence
        # repeat_aspect = RepeatVector(max_seq_len)(aspect_embed)
        #
        # # mask after concatenate will be same as hidden_out's mask
        # concat = concatenate([hidden_vecs, repeat_aspect], axis=-1)
        #
        # # apply attention mechanism
        # attend_weight = Attention(W_regularizer=regularizers.l2(0.01),
        #                             u_regularizer=regularizers.l2(0.01))(concat)
        # attend_hidden = generate_attention_output_by_attention_weight(hidden_vecs, attend_weight)

        attend_weight = Lambda(ac_aoa)([aspect_embed, hidden_vecs])
        attend_hidden = generate_attention_output_by_attention_weight(hidden_vecs, attend_weight)

        sentiment_dense_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                            bias_regularizer=regularizers.l2(0.01))(attend_hidden)
        output_layer = Dense(sentiment_class_num, activation='softmax',
                          kernel_regularizer=regularizers.l2(0.1),
                          bias_regularizer=regularizers.l2(0.01))(sentiment_dense_output)

        outputs.append(output_layer)
        loss.append(losses.categorical_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[1])

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file=data_path.data_base_dir + 'semeval_2014_task_4_rest_acaoa.png')
    return model


def aae_s_no_ada_no_sh_no_com(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    aspect_index_and_attention = {}

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding', mask_zero=True)(word_input)

    # aspect category分类
    text_embed = SpatialDropout1D(0.5, name='SpatialDropout1D_embedding')(word_embedding)

    hidden_vecs= Bidirectional(LSTM(100, return_sequences=True,
                                                           return_state=False,
                                                           name='gru',
                                                           # kernel_regularizer=regularizers.l2(0.01),
                                                           # bias_regularizer=regularizers.l2(0.01)
                                                                ))(text_embed)
    hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D_rnn')(hidden_vecs)

    for i in range(aspect_class_num):
        # apply attention mechanism
        attend_weight = Attention(use_W=False, W_regularizer=regularizers.l2(0.01),
                                  u_regularizer=regularizers.l2(0.01))(hidden_vecs)
        aspect_att_i = generate_attention_output_by_attention_weight(hidden_vecs, attend_weight)

        final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(aspect_att_i)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(final_output)
        outputs.append(aspect_output)
        loss.append(losses.binary_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[0])

    asp_embedding = Embedding(input_dim=aspect_class_num + 1, output_dim=200, trainable=True)
    for i in range(aspect_class_num):
        input_aspect = Input(shape=(1,), )
        inputs.append(input_aspect)
        aspect_embed = asp_embedding(input_aspect)
        # reshape to 2d
        aspect_embed = Flatten()(aspect_embed)
        # repeat aspect for every word in sequence
        # repeat_aspect = RepeatVector(max_seq_len)(aspect_embed)
        #
        # # mask after concatenate will be same as hidden_out's mask
        # concat = concatenate([hidden_vecs, repeat_aspect], axis=-1)
        #
        # # apply attention mechanism
        # attend_weight = Attention(W_regularizer=regularizers.l2(0.01),
        #                             u_regularizer=regularizers.l2(0.01))(concat)
        # attend_hidden = generate_attention_output_by_attention_weight(hidden_vecs, attend_weight)

        attend_weight = Lambda(ac_aoa)([aspect_embed, hidden_vecs])
        attend_hidden = generate_attention_output_by_attention_weight(hidden_vecs, attend_weight)

        sentiment_dense_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                            bias_regularizer=regularizers.l2(0.01))(attend_hidden)
        output_layer = Dense(sentiment_class_num, activation='softmax',
                          kernel_regularizer=regularizers.l2(0.1),
                          bias_regularizer=regularizers.l2(0.01))(sentiment_dense_output)

        outputs.append(output_layer)
        loss.append(losses.categorical_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[1])

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file=data_path.data_base_dir + 'semeval_2014_task_4_rest_acaoa.png')
    return model


def acaoa_block_semeval_2016_task_5_rest_sb2(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    aspect_index_and_attention = {}

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding', mask_zero=True)(word_input)

    # aspect category分类
    text_embed = SpatialDropout1D(0.5, name='SpatialDropout1D_embedding')(word_embedding)

    for i in range(aspect_class_num):
        aspect_att, sentiment_att = aae_aoa_block(text_embed)
        if i not in aspect_index_and_attention:
            aspect_index_and_attention[i] = []
        aspect_index_and_attention[i].append((aspect_att, sentiment_att))

    hidden_vecs= Bidirectional(LSTM(100, return_sequences=True,
                                                           return_state=False,
                                                           name='gru',
                                                           # kernel_regularizer=regularizers.l2(0.01),
                                                           # bias_regularizer=regularizers.l2(0.01)
                                                                ))(text_embed)
    hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D_rnn')(hidden_vecs)

    for i in range(aspect_class_num):
        aspect_att, sentiment_att = aae_aoa_block(hidden_vecs)
        if i not in aspect_index_and_attention:
            aspect_index_and_attention[i] = []
        aspect_index_and_attention[i].append((aspect_att, sentiment_att))

    aspect_attentions = []
    for i in range(aspect_class_num):
        # apply attention mechanism
        i_aspect_attentions = [a[0] for a in aspect_index_and_attention[i]]
        if len(i_aspect_attentions) == 1:
            aspect_att_i = i_aspect_attentions[0]
        else:
            aspect_att_i = concatenate(i_aspect_attentions)

        aspect_attentions.append(aspect_att_i)
        final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(aspect_att_i)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(final_output)

        outputs.append(aspect_output)
        loss.append(losses.binary_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[0])

    sentiment_dense = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                            bias_regularizer=regularizers.l2(0.01))
    sentiment_out = Dense(sentiment_class_num, activation='softmax',
                          kernel_regularizer=regularizers.l2(0.1),
                          bias_regularizer=regularizers.l2(0.01))
    for i in range(aspect_class_num):
        i_sentiment_attentions = [a[1] for a in aspect_index_and_attention[i]]
        if len(i_sentiment_attentions) == 1:
            attend_hidden = i_aspect_attentions[0]
        else:
            attend_hidden = concatenate(i_aspect_attentions)

        sentiment_dense_output = sentiment_dense(attend_hidden)
        output_layer = sentiment_out(sentiment_dense_output)

        outputs.append(output_layer)
        loss.append(losses.categorical_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[1])

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file=data_path.data_base_dir + 'semeval_2014_task_4_rest_acaoa.png')
    return model


def acaoa_block_semeval_2016_task_5_rest_sb1(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    aspect_index_and_attention = {}

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding', mask_zero=True)(word_input)

    # aspect category分类
    text_embed = SpatialDropout1D(0.5, name='SpatialDropout1D_embedding')(word_embedding)

    for i in range(aspect_class_num):
        aspect_att, sentiment_att = aae_aoa_block(text_embed)
        if i not in aspect_index_and_attention:
            aspect_index_and_attention[i] = []
        aspect_index_and_attention[i].append((aspect_att, sentiment_att))

    hidden_vecs= Bidirectional(LSTM(100, return_sequences=True,
                                                           return_state=False,
                                                           name='gru',
                                                           # kernel_regularizer=regularizers.l2(0.01),
                                                           # bias_regularizer=regularizers.l2(0.01)
                                                                ))(text_embed)
    hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D_rnn')(hidden_vecs)

    for i in range(aspect_class_num):
        aspect_att, sentiment_att = aae_aoa_block(hidden_vecs)
        if i not in aspect_index_and_attention:
            aspect_index_and_attention[i] = []
        aspect_index_and_attention[i].append((aspect_att, sentiment_att))

    aspect_attentions = []
    for i in range(aspect_class_num):
        # apply attention mechanism
        i_aspect_attentions = [a[0] for a in aspect_index_and_attention[i]]
        if len(i_aspect_attentions) == 1:
            aspect_att_i = i_aspect_attentions[0]
        else:
            aspect_att_i = concatenate(i_aspect_attentions)

        aspect_attentions.append(aspect_att_i)
        final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(aspect_att_i)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(final_output)

        outputs.append(aspect_output)
        loss.append(losses.binary_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[0])

    sentiment_dense = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                            bias_regularizer=regularizers.l2(0.01))
    sentiment_out = Dense(sentiment_class_num, activation='softmax',
                          kernel_regularizer=regularizers.l2(0.1),
                          bias_regularizer=regularizers.l2(0.01))
    for i in range(aspect_class_num):
        i_sentiment_attentions = [a[1] for a in aspect_index_and_attention[i]]
        if len(i_sentiment_attentions) == 1:
            attend_hidden = i_aspect_attentions[0]
        else:
            attend_hidden = concatenate(i_aspect_attentions)

        sentiment_dense_output = sentiment_dense(attend_hidden)
        output_layer = sentiment_out(sentiment_dense_output)

        outputs.append(output_layer)
        loss.append(losses.categorical_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[1])

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file=data_path.data_base_dir + 'semeval_2014_task_4_rest_acaoa.png')
    return model


def acaoa_semeval_2016_task_5_lapt_sb2(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding', mask_zero=True)(word_input)

    # aspect category分类
    text_embed = SpatialDropout1D(0.5, name='SpatialDropout1D_embedding')(word_embedding)
    hidden_vecs= Bidirectional(LSTM(100, return_sequences=True,
                                                           return_state=False,
                                                           name='gru',
                                                           # kernel_regularizer=regularizers.l2(0.01),
                                                           # bias_regularizer=regularizers.l2(0.01)
                                                                ))(text_embed)
    hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D_rnn')(hidden_vecs)
    aspect_attentions = []
    for i in range(aspect_class_num):
        # apply attention mechanism
        attend_weight = Attention(use_W=False, W_regularizer=regularizers.l2(0.01),
                                    u_regularizer=regularizers.l2(0.01))(hidden_vecs)
        aspect_att_i = generate_attention_output_by_attention_weight(hidden_vecs, attend_weight)

        aspect_attentions.append(aspect_att_i)
        final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(aspect_att_i)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(final_output)

        outputs.append(aspect_output)
        loss.append(losses.binary_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[0])

    hidden_vecs_sentiment = hidden_vecs
    sentiment_dense = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                            bias_regularizer=regularizers.l2(0.01))
    sentiment_out = Dense(sentiment_class_num, activation='softmax',
                          kernel_regularizer=regularizers.l2(0.1),
                          bias_regularizer=regularizers.l2(0.01))
    for i in range(aspect_class_num):
        aspect_attention = aspect_attentions[i]
        attend_weight = Lambda(ac_aoa)([aspect_attention, hidden_vecs_sentiment])
        attend_hidden = generate_attention_output_by_attention_weight(hidden_vecs_sentiment, attend_weight)

        sentiment_dense_output = sentiment_dense(attend_hidden)
        output_layer = sentiment_out(sentiment_dense_output)

        outputs.append(output_layer)
        loss.append(losses.categorical_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[1])

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file=data_path.data_base_dir + 'semeval_2014_task_4_rest_acaoa.png')
    return model


def acaoa_semeval_2016_task_5_lapt_sb1(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding', mask_zero=True)(word_input)

    # aspect category分类
    text_embed = SpatialDropout1D(0.5, name='SpatialDropout1D_embedding')(word_embedding)
    hidden_vecs= Bidirectional(LSTM(100, return_sequences=True,
                                                           return_state=False,
                                                           name='gru',
                                                           # kernel_regularizer=regularizers.l2(0.01),
                                                           # bias_regularizer=regularizers.l2(0.01)
                                                                ))(text_embed)
    hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D_rnn')(hidden_vecs)
    aspect_attentions = []
    for i in range(aspect_class_num):
        # apply attention mechanism
        attend_weight = Attention(use_W=False, W_regularizer=regularizers.l2(0.01),
                                    u_regularizer=regularizers.l2(0.01))(hidden_vecs)
        aspect_att_i = generate_attention_output_by_attention_weight(hidden_vecs, attend_weight)

        aspect_attentions.append(aspect_att_i)
        final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(aspect_att_i)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(final_output)

        outputs.append(aspect_output)
        loss.append(losses.binary_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[0])

    hidden_vecs_sentiment = hidden_vecs
    sentiment_dense = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                            bias_regularizer=regularizers.l2(0.01))
    sentiment_out = Dense(sentiment_class_num, activation='softmax',
                          kernel_regularizer=regularizers.l2(0.1),
                          bias_regularizer=regularizers.l2(0.01))
    for i in range(aspect_class_num):
        aspect_attention = aspect_attentions[i]
        attend_weight = Lambda(ac_aoa)([aspect_attention, hidden_vecs_sentiment])
        attend_hidden = generate_attention_output_by_attention_weight(hidden_vecs_sentiment, attend_weight)

        sentiment_dense_output = sentiment_dense(attend_hidden)
        output_layer = sentiment_out(sentiment_dense_output)

        outputs.append(output_layer)
        loss.append(losses.categorical_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[1])

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file=data_path.data_base_dir + 'semeval_2014_task_4_rest_acaoa.png')
    return model


def acaoa_semeval_2016_task_5_ch_phns_sb1(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding', mask_zero=True)(word_input)

    # aspect category分类
    text_embed = SpatialDropout1D(0.5, name='SpatialDropout1D_embedding')(word_embedding)
    hidden_vecs= Bidirectional(LSTM(100, return_sequences=True,
                                                           return_state=False,
                                                           name='gru',
                                                           # kernel_regularizer=regularizers.l2(0.01),
                                                           # bias_regularizer=regularizers.l2(0.01)
                                                                ))(text_embed)
    hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D_rnn')(hidden_vecs)
    aspect_attentions = []
    for i in range(aspect_class_num):
        # apply attention mechanism
        attend_weight = Attention(use_W=False, W_regularizer=regularizers.l2(0.01),
                                    u_regularizer=regularizers.l2(0.01))(hidden_vecs)
        aspect_att_i = generate_attention_output_by_attention_weight(hidden_vecs, attend_weight)

        aspect_attentions.append(aspect_att_i)
        final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(aspect_att_i)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(final_output)

        outputs.append(aspect_output)
        loss.append(losses.binary_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[0])

    hidden_vecs_sentiment = hidden_vecs
    sentiment_dense = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                            bias_regularizer=regularizers.l2(0.01))
    sentiment_out = Dense(sentiment_class_num, activation='softmax',
                          kernel_regularizer=regularizers.l2(0.1),
                          bias_regularizer=regularizers.l2(0.01))
    for i in range(aspect_class_num):
        aspect_attention = aspect_attentions[i]
        attend_weight = Lambda(ac_aoa)([aspect_attention, hidden_vecs_sentiment])
        attend_hidden = generate_attention_output_by_attention_weight(hidden_vecs_sentiment, attend_weight)

        sentiment_dense_output = sentiment_dense(attend_hidden)
        output_layer = sentiment_out(sentiment_dense_output)

        outputs.append(output_layer)
        loss.append(losses.categorical_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[1])

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file=data_path.data_base_dir + 'semeval_2014_task_4_rest_acaoa.png')
    return model


def ac_at_lstm_semeval_2014_task_4_rest(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding', mask_zero=True)(word_input)

    # aspect category分类
    text_embed = SpatialDropout1D(0.5, name='SpatialDropout1D_embedding')(word_embedding)
    hidden_vecs= Bidirectional(LSTM(100, return_sequences=True,
                                                           return_state=False,
                                                           name='gru',
                                                           # kernel_regularizer=regularizers.l2(0.01),
                                                           # bias_regularizer=regularizers.l2(0.01)
                                                                ))(text_embed)
    hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D_rnn')(hidden_vecs)
    aspect_attentions = []
    for i in range(aspect_class_num):
        # apply attention mechanism
        attend_weight = Attention(W_regularizer=regularizers.l2(0.01),
                                    u_regularizer=regularizers.l2(0.01))(hidden_vecs)
        aspect_att_i = generate_attention_output_by_attention_weight(hidden_vecs, attend_weight)

        aspect_attentions.append(aspect_att_i)
        final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(aspect_att_i)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(final_output)

        outputs.append(aspect_output)
        loss.append(losses.binary_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[0])

    hidden_vecs_sentiment = hidden_vecs
    sentiment_attention = Attention()
    sentiment_dense = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                            bias_regularizer=regularizers.l2(0.01))
    sentiment_out = Dense(sentiment_class_num, activation='softmax',
                          kernel_regularizer=regularizers.l2(0.1),
                          bias_regularizer=regularizers.l2(0.01))
    for i in range(aspect_class_num):
        aspect_attention = aspect_attentions[i]
        repeat_aspect = RepeatVector(max_seq_len)(aspect_attention)

        concat = concatenate([hidden_vecs_sentiment, repeat_aspect], axis=-1)
        attend_weight = sentiment_attention(concat)
        attend_hidden = generate_attention_output_by_attention_weight(hidden_vecs_sentiment, attend_weight)

        sentiment_dense_output = sentiment_dense(attend_hidden)
        output_layer = sentiment_out(sentiment_dense_output)

        outputs.append(output_layer)
        loss.append(losses.categorical_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[1])

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam()
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file=data_path.data_base_dir + 'semeval_2014_task_4_rest_acaoa.png')
    return model


def ac_ian_semeval_2014_task_4_rest(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding', mask_zero=True)(word_input)

    # aspect category分类
    text_embed = SpatialDropout1D(0.5, name='SpatialDropout1D_embedding')(word_embedding)
    hidden_vecs= Bidirectional(LSTM(100, return_sequences=True,
                                                           return_state=False,
                                                           name='gru',
                                                           # kernel_regularizer=regularizers.l2(0.01),
                                                           # bias_regularizer=regularizers.l2(0.01)
                                                                ))(text_embed)
    hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D_rnn')(hidden_vecs)
    aspect_attentions = []
    for i in range(aspect_class_num):
        # apply attention mechanism
        attend_weight = Attention(W_regularizer=regularizers.l2(0.01),
                                    u_regularizer=regularizers.l2(0.01))(hidden_vecs)
        aspect_att_i = generate_attention_output_by_attention_weight(hidden_vecs, attend_weight)

        weight_hidden = generate_attention_output_sequence_by_attention_weight(hidden_vecs, attend_weight)
        aspect_attentions.append(weight_hidden)

        final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(aspect_att_i)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(final_output)

        outputs.append(aspect_output)
        loss.append(losses.binary_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[0])

    hidden_vecs_sentiment = hidden_vecs
    ian = InteractiveAttention(regularizer=regularizers.l2(0.01))
    sentiment_dense = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                            bias_regularizer=regularizers.l2(0.01))
    sentiment_out = Dense(sentiment_class_num, activation='softmax',
                          kernel_regularizer=regularizers.l2(0.1),
                          bias_regularizer=regularizers.l2(0.01))
    for i in range(aspect_class_num):
        aspect_attention = aspect_attentions[i]

        attend_concat = ian([hidden_vecs_sentiment, aspect_attention])

        sentiment_dense_output = sentiment_dense(attend_concat)
        output_layer = sentiment_out(sentiment_dense_output)

        outputs.append(output_layer)
        loss.append(losses.categorical_crossentropy)
        loss_weights.append(aspect_and_sentiment_weight[1])

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam()
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file=data_path.data_base_dir + 'semeval_2014_task_4_rest_acaoa.png')
    return model


def end_to_end_lstm_semeval_2014_task_4_rest(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                               mask_zero=True)(word_input)
    text_embed = Dropout(0.5)(word_embedding)
    bilstm_result = Bidirectional(LSTM(100))(text_embed)
    bilstm_result = Dropout(0.5)(bilstm_result)
    for i in range(aspect_class_num):
        aspect_output = Dense(sentiment_class_num + 1, activation="softmax")(bilstm_result)
        outputs.append(aspect_output)
        loss.append(losses.categorical_crossentropy)

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file=data_path.data_base_dir + 'semeval_2014_task_4_rest_acaoa.png')
    return model


def end_to_end_lstm(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                               mask_zero=True)(word_input)
    text_embed = Dropout(0.5)(word_embedding)
    bilstm_result = Bidirectional(LSTM(100))(text_embed)
    bilstm_result = Dropout(0.5)(bilstm_result)
    for i in range(aspect_class_num):
        aspect_output = Dense(sentiment_class_num + 1, activation="softmax")(bilstm_result)
        outputs.append(aspect_output)
        loss.append(losses.categorical_crossentropy)

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file=data_path.data_base_dir + 'semeval_2014_task_4_rest_acaoa.png')
    return model


def end_to_end_cnn_semeval_2016_task_5_ch_came_sb1(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=True,
                               mask_zero=False)(word_input)

    convs = []
    filter_sizes = [3, 4, 5]

    for filter_size in filter_sizes:
        conv = Conv1D(filters=300, kernel_size=filter_size, activation='relu')(word_embedding)
        max_pool = GlobalMaxPooling1D()(conv)
        convs.append(max_pool)

    l_merge = concatenate(convs)

    l_merge = Dropout(0.5)(l_merge)
    for i in range(aspect_class_num):

        aspect_output = Dense(sentiment_class_num + 1, activation="softmax")(l_merge)

        outputs.append(aspect_output)
        loss.append(losses.categorical_crossentropy)

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file=data_path.data_base_dir + 'semeval_2014_task_4_rest_acaoa.png')
    return model


def end_to_end_cnn_semeval_2016_task_5_lapt_sb2(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=True,
                               mask_zero=False)(word_input)

    convs = []
    filter_sizes = [3, 4, 5]

    for filter_size in filter_sizes:
        conv = Conv1D(filters=300, kernel_size=filter_size, activation='relu')(word_embedding)
        max_pool = GlobalMaxPooling1D()(conv)
        convs.append(max_pool)

    l_merge = concatenate(convs)

    l_merge = Dropout(0.5)(l_merge)
    for i in range(aspect_class_num):

        aspect_output = Dense(sentiment_class_num + 1, activation="softmax")(l_merge)

        outputs.append(aspect_output)
        loss.append(losses.categorical_crossentropy)

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file=data_path.data_base_dir + 'semeval_2014_task_4_rest_acaoa.png')
    return model


def end_to_end_lstm_semeval_2016_task_5_ch_came_sb1(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                               mask_zero=True)(word_input)
    text_embed = Dropout(0.5)(word_embedding)
    bilstm_result = Bidirectional(LSTM(100))(text_embed)
    bilstm_result = Dropout(0.5)(bilstm_result)
    for i in range(aspect_class_num):
        aspect_output = Dense(sentiment_class_num + 1, activation="softmax")(bilstm_result)
        outputs.append(aspect_output)
        loss.append(losses.categorical_crossentropy)

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file=data_path.data_base_dir + 'semeval_2014_task_4_rest_acaoa.png')
    return model


def end_to_end_lstm_semeval_2016_task_5_lapt_sb2(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                               mask_zero=True)(word_input)
    text_embed = Dropout(0.5)(word_embedding)
    bilstm_result = Bidirectional(LSTM(100))(text_embed)
    bilstm_result = Dropout(0.5)(bilstm_result)
    for i in range(aspect_class_num):
        aspect_output = Dense(sentiment_class_num + 1, activation="softmax")(bilstm_result)
        outputs.append(aspect_output)
        loss.append(losses.categorical_crossentropy)

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file=data_path.data_base_dir + 'semeval_2014_task_4_rest_acaoa.png')
    return model


def end_to_end_lstm_semeval_2016_task_5_ch_phns_sb1(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                               mask_zero=True)(word_input)
    text_embed = Dropout(0.5)(word_embedding)
    bilstm_result = Bidirectional(LSTM(100))(text_embed)
    bilstm_result = Dropout(0.5)(bilstm_result)
    for i in range(aspect_class_num):
        aspect_output = Dense(sentiment_class_num + 1, activation="softmax")(bilstm_result)
        outputs.append(aspect_output)
        loss.append(losses.categorical_crossentropy)

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file=data_path.data_base_dir + 'semeval_2014_task_4_rest_acaoa.png')
    return model


def end_to_end_cnn_semeval_2016_task_5_ch_phns_sb1(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=True,
                               mask_zero=False)(word_input)

    convs = []
    filter_sizes = [3, 4, 5]

    for filter_size in filter_sizes:
        conv = Conv1D(filters=300, kernel_size=filter_size, activation='relu')(word_embedding)
        max_pool = GlobalMaxPooling1D()(conv)
        convs.append(max_pool)

    l_merge = concatenate(convs)

    l_merge = Dropout(0.5)(l_merge)
    for i in range(aspect_class_num):

        aspect_output = Dense(sentiment_class_num + 1, activation="softmax")(l_merge)

        outputs.append(aspect_output)
        loss.append(losses.categorical_crossentropy)

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file=data_path.data_base_dir + 'semeval_2014_task_4_rest_acaoa.png')
    return model


def end_to_end_cnn_semeval_2014_task_4_rest(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=True,
                               mask_zero=False)(word_input)

    convs = []
    filter_sizes = [3, 4, 5]

    for filter_size in filter_sizes:
        conv = Conv1D(filters=300, kernel_size=filter_size, activation='relu')(word_embedding)
        max_pool = GlobalMaxPooling1D()(conv)
        convs.append(max_pool)

    l_merge = concatenate(convs)

    l_merge = Dropout(0.5)(l_merge)
    for i in range(aspect_class_num):

        aspect_output = Dense(sentiment_class_num + 1, activation="softmax")(l_merge)

        outputs.append(aspect_output)
        loss.append(losses.categorical_crossentropy)

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file=data_path.data_base_dir + 'semeval_2014_task_4_rest_acaoa.png')
    return model


def end_to_end_cnn(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=True,
                               mask_zero=False)(word_input)

    convs = []
    filter_sizes = [3, 4, 5]

    for filter_size in filter_sizes:
        conv = Conv1D(filters=300, kernel_size=filter_size, activation='relu')(word_embedding)
        max_pool = GlobalMaxPooling1D()(conv)
        convs.append(max_pool)

    l_merge = concatenate(convs)

    l_merge = Dropout(0.5)(l_merge)
    for i in range(aspect_class_num):

        aspect_output = Dense(sentiment_class_num + 1, activation="softmax")(l_merge)

        outputs.append(aspect_output)
        loss.append(losses.categorical_crossentropy)

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file=data_path.data_base_dir + 'semeval_2014_task_4_rest_acaoa.png')
    return model


def end_to_end_attention_semeval_2014_task_4_rest(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                               mask_zero=True)(word_input)
    text_embed = Dropout(0.5)(word_embedding)
    bilstm_result = Bidirectional(LSTM(100, return_sequences=True))(text_embed)
    hidden_vecs = Dropout(0.5)(bilstm_result)
    for i in range(aspect_class_num):
        # apply attention mechanism
        attend_weight = Attention(W_regularizer=regularizers.l2(0.01),
                                    u_regularizer=regularizers.l2(0.01))(hidden_vecs)
        aspect_att_i = generate_attention_output_by_attention_weight(hidden_vecs, attend_weight)

        aspect_output = Dense(sentiment_class_num + 1, activation="sigmoid")(aspect_att_i)

        outputs.append(aspect_output)
        loss.append(losses.categorical_crossentropy)

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file=data_path.data_base_dir + 'semeval_2014_task_4_rest_acaoa.png')
    return model


def end_to_end_attention(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                               mask_zero=True)(word_input)
    text_embed = Dropout(0.5)(word_embedding)
    bilstm_result = Bidirectional(LSTM(100, return_sequences=True))(text_embed)
    hidden_vecs = Dropout(0.5)(bilstm_result)
    for i in range(aspect_class_num):
        # apply attention mechanism
        attend_weight = Attention(W_regularizer=regularizers.l2(0.01),
                                    u_regularizer=regularizers.l2(0.01))(hidden_vecs)
        aspect_att_i = generate_attention_output_by_attention_weight(hidden_vecs, attend_weight)

        aspect_output = Dense(sentiment_class_num + 1, activation="sigmoid")(aspect_att_i)

        outputs.append(aspect_output)
        loss.append(losses.categorical_crossentropy)

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file=data_path.data_base_dir + 'semeval_2014_task_4_rest_acaoa.png')
    return model


def end_to_end_attention_semeval_2016_task_5_ch_came_sb1(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                               mask_zero=True)(word_input)
    text_embed = Dropout(0.5)(word_embedding)
    bilstm_result = Bidirectional(LSTM(100, return_sequences=True))(text_embed)
    hidden_vecs = Dropout(0.5)(bilstm_result)
    for i in range(aspect_class_num):
        # apply attention mechanism
        attend_weight = Attention(use_W=False, W_regularizer=regularizers.l2(0.01),
                                    u_regularizer=regularizers.l2(0.01))(hidden_vecs)
        aspect_att_i = generate_attention_output_by_attention_weight(hidden_vecs, attend_weight)

        aspect_output = Dense(sentiment_class_num + 1, activation="sigmoid")(aspect_att_i)

        outputs.append(aspect_output)
        loss.append(losses.categorical_crossentropy)

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file=data_path.data_base_dir + 'semeval_2014_task_4_rest_acaoa.png')
    return model


def end_to_end_attention_semeval_2016_task_5_ch_phns_sb1(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                               mask_zero=True)(word_input)
    text_embed = Dropout(0.5)(word_embedding)
    bilstm_result = Bidirectional(LSTM(100, return_sequences=True))(text_embed)
    hidden_vecs = Dropout(0.5)(bilstm_result)
    for i in range(aspect_class_num):
        # apply attention mechanism
        attend_weight = Attention(use_W=False, W_regularizer=regularizers.l2(0.01),
                                    u_regularizer=regularizers.l2(0.01))(hidden_vecs)
        aspect_att_i = generate_attention_output_by_attention_weight(hidden_vecs, attend_weight)

        aspect_output = Dense(sentiment_class_num + 1, activation="sigmoid")(aspect_att_i)

        outputs.append(aspect_output)
        loss.append(losses.categorical_crossentropy)

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file=data_path.data_base_dir + 'semeval_2014_task_4_rest_acaoa.png')
    return model


def end_to_end_attention_semeval_2016_task_5_lapt_sb2(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                               mask_zero=True)(word_input)
    text_embed = Dropout(0.5)(word_embedding)
    bilstm_result = Bidirectional(LSTM(100, return_sequences=True))(text_embed)
    hidden_vecs = Dropout(0.5)(bilstm_result)
    for i in range(aspect_class_num):
        # apply attention mechanism
        attend_weight = Attention(use_W=False, W_regularizer=regularizers.l2(0.01),
                                    u_regularizer=regularizers.l2(0.01))(hidden_vecs)
        aspect_att_i = generate_attention_output_by_attention_weight(hidden_vecs, attend_weight)

        aspect_output = Dense(sentiment_class_num + 1, activation="sigmoid")(aspect_att_i)

        outputs.append(aspect_output)
        loss.append(losses.categorical_crossentropy)

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(clipvalue=5)
    model.compile(loss=loss,
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file=data_path.data_base_dir + 'semeval_2014_task_4_rest_acaoa.png')
    return model


def bdci2018_acaoa(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1, 0.001]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding', mask_zero=True)(word_input)

    # aspect category分类
    text_embed = SpatialDropout1D(0.5, name='SpatialDropout1D_embedding')(word_embedding)
    hidden_vecs, last_state_0, last_state_1 = Bidirectional(GRU(150, return_sequences=True,
                                                           return_state=True,
                                                           name='gru',
                                                           # kernel_regularizer=regularizers.l2(0.01),
                                                           # bias_regularizer=regularizers.l2(0.01)
                                                                ))(text_embed)
    hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D_rnn')(hidden_vecs)
    aspect_attentions = []
    aspect_outputs = []
    for i in range(aspect_class_num):
        # apply attention mechanism
        attend_weight = Attention(W_regularizer=regularizers.l2(0.01),
                                    u_regularizer=regularizers.l2(0.01))(hidden_vecs)
        attend_weight_expand = Lambda(lambda x: K.expand_dims(x))(attend_weight)
        attend_hidden = multiply([hidden_vecs, attend_weight_expand])
        aspect_att_i = Lambda(lambda x: K.sum(x, axis=1))(attend_hidden)

        aspect_attentions.append(aspect_att_i)
        final_output = Dense(32, activation='relu')(aspect_att_i)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01))(final_output)

        aspect_output_no_grad = Lambda(lambda x: K.stop_gradient(x))(aspect_output)
        aspect_outputs.append(aspect_output_no_grad)

        outputs.append(aspect_output)
        loss.append('binary_crossentropy')
        loss_weights.append(aspect_and_sentiment_weight[0])

    hidden_vecs_sentiment = hidden_vecs
    sentiment_dense = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))
    sentiment_out = Dense(sentiment_class_num, activation='softmax',
                          kernel_regularizer=regularizers.l2(0.1))
    for i in range(aspect_class_num):
        aspect_attention = aspect_attentions[i]
        attend_weight = Lambda(ac_aoa)([aspect_attention, hidden_vecs_sentiment])

        attend_weight_expand = Lambda(lambda x: K.expand_dims(x))(attend_weight)
        attend_hidden = multiply([hidden_vecs_sentiment, attend_weight_expand])
        attend_hidden = Lambda(lambda x: K.sum(x, axis=1))(attend_hidden)
        sentiment_dense_output = sentiment_dense(attend_hidden)
        output_layer = sentiment_out(sentiment_dense_output)
        aspect_output = aspect_outputs[i]
        weight_sentiment_output = multiply([output_layer, aspect_output])
        outputs.append(weight_sentiment_output)
        loss.append('categorical_crossentropy')
        loss_weights.append(aspect_and_sentiment_weight[1])

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(lr=loss_weights[2])
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file=data_path.data_base_dir + 'semeval_2014_task_4_rest_acaoa.png')
    return model


def semeval_141516_rest_acaoa(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding', mask_zero=True)(word_input)

    # aspect category分类
    text_embed = SpatialDropout1D(0.5, name='SpatialDropout1D_embedding')(word_embedding)
    hidden_vecs, last_state_0, last_state_1 = Bidirectional(GRU(75, return_sequences=True,
                                                           return_state=True,
                                                           name='gru',
                                                           # kernel_regularizer=regularizers.l2(0.01),
                                                           # bias_regularizer=regularizers.l2(0.01)
                                                                ))(text_embed)
    hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D_rnn')(hidden_vecs)
    aspect_attentions = []
    aspect_outputs = []
    for i in range(aspect_class_num):
        # apply attention mechanism
        attend_weight = Attention(W_regularizer=regularizers.l2(0.01),
                                    u_regularizer=regularizers.l2(0.01))(hidden_vecs)
        attend_weight_expand = Lambda(lambda x: K.expand_dims(x))(attend_weight)
        attend_hidden = multiply([hidden_vecs, attend_weight_expand])
        aspect_att_i = Lambda(lambda x: K.sum(x, axis=1))(attend_hidden)

        aspect_attentions.append(aspect_att_i)
        final_output = Dense(32, activation='relu')(aspect_att_i)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01))(final_output)

        aspect_output_no_grad = Lambda(lambda x: K.stop_gradient(x))(aspect_output)
        aspect_outputs.append(aspect_output_no_grad)

        outputs.append(aspect_output)
        loss.append('binary_crossentropy')
        loss_weights.append(aspect_and_sentiment_weight[0])

    hidden_vecs_sentiment = hidden_vecs
    sentiment_dense = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))
    sentiment_out = Dense(sentiment_class_num, activation='softmax',
                          kernel_regularizer=regularizers.l2(0.1))
    for i in range(aspect_class_num):
        aspect_attention = aspect_attentions[i]
        attend_weight = Lambda(ac_aoa)([aspect_attention, hidden_vecs_sentiment])

        attend_weight_expand = Lambda(lambda x: K.expand_dims(x))(attend_weight)
        attend_hidden = multiply([hidden_vecs_sentiment, attend_weight_expand])
        attend_hidden = Lambda(lambda x: K.sum(x, axis=1))(attend_hidden)
        sentiment_dense_output = sentiment_dense(attend_hidden)
        output_layer = sentiment_out(sentiment_dense_output)
        aspect_output = aspect_outputs[i]
        weight_sentiment_output = multiply([output_layer, aspect_output])
        outputs.append(weight_sentiment_output)
        loss.append('categorical_crossentropy')
        loss_weights.append(aspect_and_sentiment_weight[1])

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam()
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file=data_path.data_base_dir + 'semeval_2014_task_4_rest_acaoa.png')
    return model


from keras_self_attention import SeqSelfAttention


def semeval_2014_task_4_rest_acaoa_transformer(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3,
                                   aspect_and_sentiment_weight: list=[1, 1]):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding', mask_zero=True)(word_input)

    # aspect category分类
    text_embed = SpatialDropout1D(0.5, name='SpatialDropout1D_embedding')(word_embedding)
    # hidden_vecs, last_state_0, last_state_1 = Bidirectional(GRU(150, return_sequences=True,
    #                                                        return_state=True,
    #                                                        name='gru',
    #                                                        # kernel_regularizer=regularizers.l2(0.01),
    #                                                        # bias_regularizer=regularizers.l2(0.01)
    #                                                             ))(text_embed)
    # embeddings = self_attention_keras.Position_Embedding()(text_embed)  # 增加Position_Embedding能轻微提高准确率
    hidden_vecs = SeqSelfAttention(attention_activation='sigmoid')(text_embed)

    hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D_rnn')(hidden_vecs)
    aspect_attentions = []
    aspect_outputs = []
    for i in range(aspect_class_num):
        # apply attention mechanism
        attend_weight = Attention(W_regularizer=regularizers.l2(0.01),
                                    u_regularizer=regularizers.l2(0.01))(hidden_vecs)
        attend_weight_expand = Lambda(lambda x: K.expand_dims(x))(attend_weight)
        attend_hidden = multiply([hidden_vecs, attend_weight_expand])
        aspect_att_i = Lambda(lambda x: K.sum(x, axis=1))(attend_hidden)

        aspect_attentions.append(aspect_att_i)
        final_output = Dense(32, activation='relu')(aspect_att_i)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01))(final_output)
        aspect_outputs.append(aspect_output)
        outputs.append(aspect_output)
        loss.append('binary_crossentropy')
        loss_weights.append(aspect_and_sentiment_weight[0])

    hidden_vecs_sentiment = hidden_vecs
    sentiment_dense = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))
    sentiment_out = Dense(sentiment_class_num, activation='softmax',
                          kernel_regularizer=regularizers.l2(0.1))
    for i in range(aspect_class_num):
        aspect_attention = aspect_attentions[i]
        attend_weight = Lambda(ac_aoa)([aspect_attention, hidden_vecs_sentiment])

        attend_weight_expand = Lambda(lambda x: K.expand_dims(x))(attend_weight)
        attend_hidden = multiply([hidden_vecs_sentiment, attend_weight_expand])
        attend_hidden = Lambda(lambda x: K.sum(x, axis=1))(attend_hidden)
        sentiment_dense_output = sentiment_dense(attend_hidden)
        output_layer = sentiment_out(sentiment_dense_output)
        aspect_output = aspect_outputs[i]
        weight_sentiment_output = multiply([output_layer, aspect_output])
        outputs.append(weight_sentiment_output)
        loss.append('categorical_crossentropy')
        loss_weights.append(aspect_and_sentiment_weight[1])

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam()
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    # plot_model(model, to_file=data_path.data_base_dir + 'semeval_2014_task_4_rest_acaoa.png')
    return model


def joint_model_of_aspect_and_sentiment_semeval2014_acaoa_without_rnn(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding', mask_zero=True)(word_input)

    # aspect category分类
    text_embed = SpatialDropout1D(0.5, name='SpatialDropout1D_embedding')(word_embedding)
    # # hidden_vecs = GRU(50, return_sequences=True)(text_embed)
    # hidden_vecs, last_state_0, last_state_1 = Bidirectional(GRU(25, return_sequences=True,
    #                                                        return_state=True,
    #                                                        name='gru',
    #                                                        # kernel_regularizer=regularizers.l2(0.01),
    #                                                        # bias_regularizer=regularizers.l2(0.01)
    #                                                             ))(text_embed)
    # hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D_rnn')(hidden_vecs)
    # hidden_vecs = Dropout(0.3, name='Dropout')(hidden_vecs)
    aspect_attentions = []
    aspect_outputs = []
    for i in range(aspect_class_num):
        # apply attention mechanism
        attend_weight = Attention(W_regularizer=regularizers.l2(0.01),
                                    u_regularizer=regularizers.l2(0.01))(text_embed)
        attend_weight_expand = Lambda(lambda x: K.expand_dims(x))(attend_weight)
        attend_hidden = multiply([text_embed, attend_weight_expand])
        aspect_att_i = Lambda(lambda x: K.sum(x, axis=1))(attend_hidden)

        # aspect_att_i = keras_layers.AttLayer(20, i)(hidden_vecs)
        aspect_attentions.append(aspect_att_i)
        final_output = Dense(32, activation='relu')(aspect_att_i)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01))(final_output)
        aspect_outputs.append(aspect_output)
        outputs.append(aspect_output)
        loss.append('binary_crossentropy')
        loss_weights.append(1.0)

    # 主题对应的情感分类
    # hidden_vecs_sentiment, last_state_0_sentiment, last_state_1_sentiment = Bidirectional(GRU(25, return_sequences=True,
    #                                                             return_state=True,
    #                                                             name='hidden_vecs_sentiment',
    #                                                             # kernel_regularizer=regularizers.l2(0.01),
    #                                                             # bias_regularizer=regularizers.l2(0.01)
    #                                                             ))(word_embedding)
    # hidden_vecs_sentiment = SpatialDropout1D(0.5, name='SpatialDropout1D_sentiment')(hidden_vecs_sentiment)
    hidden_vecs_sentiment = text_embed
    sentiment_dense = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))
    sentiment_out = Dense(sentiment_class_num, activation='softmax',
                          kernel_regularizer=regularizers.l2(0.1))
    for i in range(aspect_class_num):
        aspect_attention = aspect_attentions[i]
        attend_weight = Lambda(ac_aoa)([aspect_attention, hidden_vecs_sentiment])

        attend_weight_expand = Lambda(lambda x: K.expand_dims(x))(attend_weight)
        attend_hidden = multiply([hidden_vecs_sentiment, attend_weight_expand])
        attend_hidden = Lambda(lambda x: K.sum(x, axis=1))(attend_hidden)
        sentiment_dense_output = sentiment_dense(attend_hidden)
        output_layer = sentiment_out(sentiment_dense_output)
        aspect_output = aspect_outputs[i]
        weight_sentiment_output = multiply([output_layer, aspect_output])
        outputs.append(weight_sentiment_output)
        loss.append('categorical_crossentropy')
        loss_weights.append(1)

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam()
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    return model


def joint_model_of_aspect_and_sentiment_semeval2016_old(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    # 共享embedding层
    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding')(word_input)

    # aspect category分类
    text_embed = SpatialDropout1D(0.5)(word_embedding)
    # hidden_vecs = GRU(50, return_sequences=True)(text_embed)
    hidden_vecs, last_state_0, last_state_1 = Bidirectional(GRU(25, return_sequences=True,
                                                           return_state=True,
                                                           name='gru',
                                                           # kernel_regularizer=regularizers.l2(0.01),
                                                           # bias_regularizer=regularizers.l2(0.01)
                                                                ))(text_embed)
    # hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D')(hidden_vecs)
    # hidden_vecs = Dropout(0.3, name='Dropout')(hidden_vecs)
    aspect_attentions = []
    for i in range(aspect_class_num):
        # apply attention mechanism
        attend_weight = Attention(use_W=True, W_regularizer=regularizers.l2(0.01),
                                    u_regularizer=regularizers.l2(0.01))(hidden_vecs)
        aspect_att_i = generate_attention_output_by_attention_weight(hidden_vecs, attend_weight)

        # aspect_att_i = keras_layers.AttLayer(20, i)(hidden_vecs)
        aspect_attentions.append(aspect_att_i)
        final_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(aspect_att_i)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01))(final_output)
        outputs.append(aspect_output)
        loss.append('binary_crossentropy')
        loss_weights.append(1)

    # 主题对应的情感分类
    hidden_vecs_sentiment, last_state_0_sentiment, last_state_1_sentiment = Bidirectional(GRU(25, return_sequences=True,
                                                                return_state=True,
                                                                name='hidden_vecs_sentiment',
                                                                # kernel_regularizer=regularizers.l2(0.01),
                                                                # bias_regularizer=regularizers.l2(0.01)
                                                                ))(word_embedding)
    hidden_vecs_sentiment = SpatialDropout1D(0.5, name='SpatialDropout1D_sentiment')(hidden_vecs_sentiment)
    sentiment_attention = Attention(W_regularizer=regularizers.l2(0.01),
                                    u_regularizer=regularizers.l2(0.01))
    asp_embedding = Embedding(input_dim=aspect_class_num + 1, output_dim=50, trainable=True)
    # sentiment_dense = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))
    sentiment_out = Dense(sentiment_class_num, activation='softmax',
                          kernel_regularizer=regularizers.l2(0.1))
    for i in range(aspect_class_num):
        input_aspect = Input(shape=(1,), )
        inputs.append(input_aspect)
        aspect_embed = asp_embedding(input_aspect)
        # reshape to 2d
        aspect_embed = Flatten()(aspect_embed)
        # repeat aspect for every word in sequence
        repeat_aspect = RepeatVector(max_seq_len)(aspect_embed)

        # mask after concatenate will be same as hidden_out's mask
        concat = concatenate([hidden_vecs_sentiment, repeat_aspect], axis=-1)

        # apply attention mechanism
        attend_weight = sentiment_attention(concat)
        attend_weight_expand = Lambda(lambda x: K.expand_dims(x))(attend_weight)
        attend_hidden = multiply([hidden_vecs_sentiment, attend_weight_expand])
        attend_hidden = Lambda(lambda x: K.sum(x, axis=1))(attend_hidden)

        # aspect_att_and_sentiment_att = Add()([aspect_attentions[i], attend_hidden])
        # aspect_att_and_sentiment_att = attend_hidden
        # aspect_att_and_sentiment_att = Dropout(0.3, name='Dropout_' + str(i))(aspect_att_and_sentiment_att)
        # dense_layer = sentiment_dense(aspect_att_and_sentiment_att)
        # aspect_att_and_sentiment_att = Dropout(0.3, name='Dropout_' + str(i))(aspect_att_and_sentiment_att)
        output_layer = sentiment_out(attend_hidden)
        outputs.append(output_layer)
        loss.append('categorical_crossentropy')
        loss_weights.append(1)

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam()
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    return model


def joint_model_of_aspect_and_sentiment_semeval2014_four(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding')(word_input)
    # text_embed = SpatialDropout1D(0.2, name='SpatialDropout1D')(word_embedding)
    # hidden_vecs = GRU(50, return_sequences=True)(text_embed)
    hidden_vecs, last_state_0, last_state_1 = Bidirectional(GRU(25, return_sequences=True,
                                                           return_state=True,
                                                           name='gru',
                                                           kernel_regularizer=regularizers.l2(0.01),
                                                           bias_regularizer=regularizers.l2(0.01)
                                                                ))(word_embedding)
    # hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D')(hidden_vecs)
    # hidden_vecs = Dropout(0.3, name='Dropout')(hidden_vecs)
    aspect_attentions = []
    for i in range(aspect_class_num):
        # apply attention mechanism
        attend_weight = Attention(W_regularizer=regularizers.l2(0.01),
                                    u_regularizer=regularizers.l2(0.01))(hidden_vecs)
        attend_weight_expand = Lambda(lambda x: K.expand_dims(x))(attend_weight)
        attend_hidden = multiply([hidden_vecs, attend_weight_expand])
        aspect_att_i = Lambda(lambda x: K.sum(x, axis=1))(attend_hidden)

        # aspect_att_i = keras_layers.AttLayer(20, i)(hidden_vecs)
        aspect_attentions.append(aspect_att_i)
        # final_output = Dense(32, activation='relu')(aspect_att_i)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01))(aspect_att_i)
        outputs.append(aspect_output)
        loss.append('binary_crossentropy')
        loss_weights.append(1)

    sentiment_attention = Attention(W_regularizer=regularizers.l2(0.01),
                                    u_regularizer=regularizers.l2(0.01))
    asp_embedding = Embedding(input_dim=6, output_dim=50, trainable=True)
    # sentiment_dense = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))
    sentiment_out = Dense(sentiment_class_num, activation='softmax',
                          kernel_regularizer=regularizers.l2(0.1))
    for i in range(aspect_class_num):
        input_aspect = Input(shape=(1,), )
        inputs.append(input_aspect)
        aspect_embed = asp_embedding(input_aspect)
        # reshape to 2d
        aspect_embed = Flatten()(aspect_embed)
        # repeat aspect for every word in sequence
        repeat_aspect = RepeatVector(max_seq_len)(aspect_embed)

        # mask after concatenate will be same as hidden_out's mask
        concat = concatenate([hidden_vecs, repeat_aspect], axis=-1)

        # apply attention mechanism
        attend_weight = sentiment_attention(concat)
        attend_weight_expand = Lambda(lambda x: K.expand_dims(x))(attend_weight)
        attend_hidden = multiply([hidden_vecs, attend_weight_expand])
        attend_hidden = Lambda(lambda x: K.sum(x, axis=1))(attend_hidden)

        # aspect_att_and_sentiment_att = Add()([aspect_attentions[i], attend_hidden])
        # aspect_att_and_sentiment_att = attend_hidden
        # aspect_att_and_sentiment_att = Dropout(0.3, name='Dropout_' + str(i))(aspect_att_and_sentiment_att)
        # dense_layer = sentiment_dense(aspect_att_and_sentiment_att)
        # aspect_att_and_sentiment_att = Dropout(0.3, name='Dropout_' + str(i))(aspect_att_and_sentiment_att)
        output_layer = sentiment_out(attend_hidden)
        outputs.append(output_layer)
        loss.append('categorical_crossentropy')
        loss_weights.append(1)

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam()
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    return model


def joint_model_of_aspect_and_sentiment_semeval2014_aspect(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3):
    """at_lstm"""
    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    word_input = Input(shape=(max_seq_len,), name='word_input')
    inputs.append(word_input)
    word_embedding = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding')(word_input)
    # text_embed = SpatialDropout1D(0.2, name='SpatialDropout1D')(word_embedding)
    # hidden_vecs = GRU(50, return_sequences=True)(text_embed)
    hidden_vecs, last_state_0, last_state_1 = Bidirectional(GRU(25, return_sequences=True,
                                                           return_state=True,
                                                           name='gru',
                                                           # kernel_regularizer=regularizers.l2(0.01),
                                                           # bias_regularizer=regularizers.l2(0.01)
                                                                ))(word_embedding)
    hidden_vecs = SpatialDropout1D(0.5, name='SpatialDropout1D')(hidden_vecs)
    # hidden_vecs = Dropout(0.3, name='Dropout')(hidden_vecs)
    aspect_attentions = []
    for i in range(aspect_class_num):
        # apply attention mechanism
        attend_weight = Attention(W_regularizer=regularizers.l2(0.01),
                                    u_regularizer=regularizers.l2(0.01))(hidden_vecs)
        attend_weight_expand = Lambda(lambda x: K.expand_dims(x))(attend_weight)
        attend_hidden = multiply([hidden_vecs, attend_weight_expand])
        aspect_att_i = Lambda(lambda x: K.sum(x, axis=1))(attend_hidden)

        # aspect_att_i = keras_layers.AttLayer(20, i)(hidden_vecs)
        aspect_attentions.append(aspect_att_i)

    sentiment_attention = Attention(W_regularizer=regularizers.l2(0.01),
                                    u_regularizer=regularizers.l2(0.01))
    asp_embedding = Embedding(input_dim=6, output_dim=50, trainable=True)
    # sentiment_dense = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))
    sentiment_out = Dense(sentiment_class_num, activation='softmax',
                          kernel_regularizer=regularizers.l2(0.1))
    sentiment_attentions = []
    for i in range(aspect_class_num):
        input_aspect = Input(shape=(1,), )
        inputs.append(input_aspect)
        aspect_embed = asp_embedding(input_aspect)
        # reshape to 2d
        aspect_embed = Flatten()(aspect_embed)
        # repeat aspect for every word in sequence
        repeat_aspect = RepeatVector(max_seq_len)(aspect_embed)

        # mask after concatenate will be same as hidden_out's mask
        concat = concatenate([hidden_vecs, repeat_aspect], axis=-1)

        # apply attention mechanism
        attend_weight = sentiment_attention(concat)
        attend_weight_expand = Lambda(lambda x: K.expand_dims(x))(attend_weight)
        attend_hidden = multiply([hidden_vecs, attend_weight_expand])
        attend_hidden = Lambda(lambda x: K.sum(x, axis=1))(attend_hidden)

        # aspect_att_and_sentiment_att = Add()([aspect_attentions[i], attend_hidden])
        # aspect_att_and_sentiment_att = attend_hidden
        # aspect_att_and_sentiment_att = Dropout(0.3, name='Dropout_' + str(i))(aspect_att_and_sentiment_att)
        # dense_layer = sentiment_dense(aspect_att_and_sentiment_att)
        # aspect_att_and_sentiment_att = Dropout(0.3, name='Dropout_' + str(i))(aspect_att_and_sentiment_att)
        sentiment_attentions.append(attend_hidden)
        output_layer = sentiment_out(attend_hidden)
        outputs.append(output_layer)
        loss.append('categorical_crossentropy')
        loss_weights.append(1)

    for i, aspect_att in enumerate(aspect_attentions):
        # final_output = Dense(32, activation='relu')(aspect_att_i)
        aspect_att_and_sentiment_att = Add()([aspect_att, aspect_attentions[i]])
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01))(
            aspect_att_and_sentiment_att)
        outputs.insert(i, aspect_output)
        loss.insert(i, 'binary_crossentropy')
        loss_weights.insert(i, 1)

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(lr=0.001)
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    return model


def joint_model_of_aspect_and_sentiment_semeval2014_bert(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3):
    """at_lstm"""

    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    config_path = data_path.bert_config_path
    checkpoint_path = data_path.bert_checkpoint_path

    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, training=False,
                                                    seq_len=max_seq_len)
    hidden_vecs = bert_model.outputs[0]
    bert_cla = keras_layers.ClassificationHiddenLayer()(hidden_vecs)

    inputs.extend(bert_model.input)

    # hidden_vecs = Dropout(0.3, name='Dropout')(hidden_vecs)
    aspect_attentions = []
    for i in range(aspect_class_num):
        # apply attention mechanism
        attend_weight = Attention(W_regularizer=regularizers.l2(0.01),
                                    u_regularizer=regularizers.l2(0.01))(hidden_vecs)
        attend_weight_expand = Lambda(lambda x: K.expand_dims(x))(attend_weight)
        attend_hidden = multiply([hidden_vecs, attend_weight_expand])
        aspect_att_i = Lambda(lambda x: K.sum(x, axis=1))(attend_hidden)

        # aspect_att_i = keras_layers.AttLayer(20, i)(hidden_vecs)
        aspect_attentions.append(aspect_att_i)
        # final_output = Dense(32, activation='relu')(aspect_att_i)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01))(aspect_att_i)
        outputs.append(aspect_output)
        loss.append('binary_crossentropy')
        loss_weights.append(1)

    sentiment_attention = Attention(W_regularizer=regularizers.l2(0.01),
                                    u_regularizer=regularizers.l2(0.01))
    asp_embedding = Embedding(input_dim=6, output_dim=50, trainable=True)
    # sentiment_dense = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))
    sentiment_out = Dense(sentiment_class_num, activation='softmax',
                          kernel_regularizer=regularizers.l2(0.1))
    for i in range(aspect_class_num):
        input_aspect = Input(shape=(1,), )
        inputs.append(input_aspect)
        aspect_embed = asp_embedding(input_aspect)
        # reshape to 2d
        aspect_embed = Flatten()(aspect_embed)
        # repeat aspect for every word in sequence
        repeat_aspect = RepeatVector(max_seq_len)(aspect_embed)

        # mask after concatenate will be same as hidden_out's mask
        concat = concatenate([hidden_vecs, repeat_aspect], axis=-1)

        # apply attention mechanism
        attend_weight = sentiment_attention(concat)
        attend_weight_expand = Lambda(lambda x: K.expand_dims(x))(attend_weight)
        attend_hidden = multiply([hidden_vecs, attend_weight_expand])
        attend_hidden = Lambda(lambda x: K.sum(x, axis=1))(attend_hidden)

        aspect_att_and_sentiment_att = Add()([aspect_attentions[i], attend_hidden])
        # aspect_att_and_sentiment_att = attend_hidden
        # aspect_att_and_sentiment_att = Dropout(0.3, name='Dropout_' + str(i))(aspect_att_and_sentiment_att)
        # dense_layer = sentiment_dense(aspect_att_and_sentiment_att)
        # aspect_att_and_sentiment_att = Dropout(0.3, name='Dropout_' + str(i))(aspect_att_and_sentiment_att)
        output_layer = sentiment_out(aspect_att_and_sentiment_att)
        outputs.append(output_layer)
        loss.append('categorical_crossentropy')
        loss_weights.append(0)

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(lr=0.001)
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    return model


def joint_model_of_aspect_and_sentiment_semeval2014_bert_four(max_seq_len, input_dim, output_dim, embedding_matrix,
                                        aspect_class_num=10, sentiment_class_num=3):
    """at_lstm"""

    inputs = []
    outputs = []
    loss = []
    loss_weights = []

    config_path = data_path.bert_config_path
    checkpoint_path = data_path.bert_checkpoint_path

    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, training=False,
                                                    seq_len=max_seq_len)
    hidden_vecs = bert_model.outputs[0]
    bert_cla = keras_layers.ClassificationHiddenLayer()(hidden_vecs)

    inputs.extend(bert_model.input)

    # hidden_vecs = Dropout(0.3, name='Dropout')(hidden_vecs)
    aspect_attentions = []
    for i in range(aspect_class_num):
        # apply attention mechanism
        attend_weight = Attention(W_regularizer=regularizers.l2(0.01),
                                    u_regularizer=regularizers.l2(0.01))(hidden_vecs)
        attend_weight_expand = Lambda(lambda x: K.expand_dims(x))(attend_weight)
        attend_hidden = multiply([hidden_vecs, attend_weight_expand])
        aspect_att_i = Lambda(lambda x: K.sum(x, axis=1))(attend_hidden)

        # aspect_att_i = keras_layers.AttLayer(20, i)(hidden_vecs)
        aspect_attentions.append(aspect_att_i)
        # final_output = Dense(32, activation='relu')(aspect_att_i)
        # aspect_att_i = Dropout(0.3, name='Dropout')(aspect_att_i)
        aspect_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01))(aspect_att_i)
        outputs.append(aspect_output)
        loss.append('binary_crossentropy')
        loss_weights.append(1)

    sentiment_attention = Attention(W_regularizer=regularizers.l2(0.01),
                                    u_regularizer=regularizers.l2(0.01))
    asp_embedding = Embedding(input_dim=6, output_dim=50, trainable=True)
    # sentiment_dense = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))
    sentiment_out = Dense(sentiment_class_num, activation='softmax',
                          kernel_regularizer=regularizers.l2(0.1))
    for i in range(aspect_class_num):
        input_aspect = Input(shape=(1,), )
        inputs.append(input_aspect)
        aspect_embed = asp_embedding(input_aspect)
        # reshape to 2d
        aspect_embed = Flatten()(aspect_embed)
        # repeat aspect for every word in sequence
        repeat_aspect = RepeatVector(max_seq_len)(aspect_embed)

        # mask after concatenate will be same as hidden_out's mask
        concat = concatenate([hidden_vecs, repeat_aspect], axis=-1)

        # apply attention mechanism
        attend_weight = sentiment_attention(concat)
        attend_weight_expand = Lambda(lambda x: K.expand_dims(x))(attend_weight)
        attend_hidden = multiply([hidden_vecs, attend_weight_expand])
        attend_hidden = Lambda(lambda x: K.sum(x, axis=1))(attend_hidden)

        # aspect_att_and_sentiment_att = Add()([aspect_attentions[i], attend_hidden])
        # aspect_att_and_sentiment_att = attend_hidden
        # aspect_att_and_sentiment_att = Dropout(0.3, name='Dropout_' + str(i))(aspect_att_and_sentiment_att)
        # dense_layer = sentiment_dense(aspect_att_and_sentiment_att)
        # aspect_att_and_sentiment_att = Dropout(0.3, name='Dropout_' + str(i))(aspect_att_and_sentiment_att)
        output_layer = sentiment_out(attend_hidden)
        outputs.append(output_layer)
        loss.append('categorical_crossentropy')
        loss_weights.append(0)

    model = Model(inputs=inputs, outputs=outputs, name='model')
    adam = optimizers.adam(lr=0.001)
    model.compile(loss=loss,
                  optimizer=adam,
                  loss_weights=loss_weights,
                  metrics=['accuracy'])
    model.summary()
    return model


def at_lstm_lstm(max_seq_len, input_dim, output_dim, embedding_matrix, num_class=10):
    """at_lstm"""
    word_input = Input(shape=(max_seq_len,), name='word_input')
    x_word = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding')(word_input)

    x_word = SpatialDropout1D(0.2, name='SpatialDropout1D')(x_word)
    x_word = Bidirectional(LSTM(60, return_sequences=True,
                               return_state=False,
                               name='gru',
                               kernel_regularizer=regularizers.l2(0.01),
                               bias_regularizer=regularizers.l2(0.01)))(x_word)

    aspect_input = Input(shape=(max_seq_len,), name='aspect_input')
    x_aspect = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                         name='aspect_embedding')(aspect_input)

    x_word_and_aspect = concatenate([x_word, x_aspect])

    x = keras_layers.AttLayerDifferentKeyValue(50, name='AttLayerDifferentKeyValue')(
        [x_word_and_aspect, x_word])
    # x = concatenate([x, last_state_0, last_state_1])
    outp = Dense(num_class, activation="softmax", name='dense', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(x)

    model = Model(inputs=[word_input, aspect_input], outputs=outp, name='model')
    adam = optimizers.adam(clipvalue=1)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    return model


def at_lstm_multi_label(max_seq_len, input_dim, output_dim, embedding_matrix, num_class=10):
    """at_lstm"""
    word_input = Input(shape=(max_seq_len,), name='word_input')
    x_word = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding')(word_input)

    # x_word = SpatialDropout1D(0.2, name='SpatialDropout1D')(x_word)
    x_word, last_state_0, last_state_1 = Bidirectional(GRU(60, return_sequences=True,
                               return_state=True,
                               name='gru',
                               kernel_regularizer=regularizers.l2(0.01),
                               bias_regularizer=regularizers.l2(0.01)))(x_word)

    aspect_input = Input(shape=(max_seq_len,), name='aspect_input')
    x_aspect = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                         name='aspect_embedding')(aspect_input)

    x_word_and_aspect = concatenate([x_word, x_aspect])

    x = keras_layers.AttLayerDifferentKeyValue(50, name='AttLayerDifferentKeyValue')(
        [x_word_and_aspect, x_word])
    # x = concatenate([x, last_state_0, last_state_1])
    outp = Dense(num_class, activation="sigmoid", name='dense', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(x)

    model = Model(inputs=[word_input, aspect_input], outputs=outp, name='model')
    adam = optimizers.adam(clipvalue=1)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    return model


def at_lstm_punctuation(max_seq_len, input_dim, output_dim, embedding_matrix, num_class=10):
    """at_lstm"""
    word_input = Input(shape=(max_seq_len,), name='word_input')
    x_word = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding')(word_input)

    x_word = SpatialDropout1D(0.2, name='SpatialDropout1D')(x_word)
    x_word = Bidirectional(GRU(60, return_sequences=True, name='gru', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01)))(x_word)

    aspect_input = Input(shape=(max_seq_len,), name='aspect_input')
    x_aspect = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                         name='aspect_embedding')(aspect_input)

    punctuation_input = Input(shape=(max_seq_len,), name='punctuation_input')
    x_punctuation = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                         name='punctuation_embedding')(punctuation_input)

    x_word_and_punctuation = concatenate([x_word, x_punctuation])

    x_word_and_punctuation_and_aspect = concatenate([x_word, x_punctuation])

    x = keras_layers.AttLayerDifferentKeyValue(50, name='AttLayerDifferentKeyValue')(
        [x_word_and_punctuation_and_aspect, x_word_and_punctuation])
    outp = Dense(num_class, activation="softmax", name='dense', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01))(x)

    model = Model(inputs=[word_input, aspect_input, punctuation_input], outputs=outp, name='model')
    adam = optimizers.adam(clipvalue=1)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    return model


def at_lstm_for_topic(max_seq_len, input_dim, output_dim, embedding_matrix, num_class=10):
    """at_lstm"""
    word_input = Input(shape=(max_seq_len,), name='word_input')
    x_word = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding')(word_input)

    x_word = SpatialDropout1D(0.2, name='SpatialDropout1D')(x_word)
    x_word = Bidirectional(GRU(20, return_sequences=True, name='gru'))(x_word)

    aspect_input = Input(shape=(max_seq_len,), name='aspect_input')
    x_aspect = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                         name='aspect_embedding')(aspect_input)

    x_word_and_aspect = concatenate([x_word, x_aspect])

    x = keras_layers.AttLayerDifferentKeyValue(20, name='AttLayerDifferentKeyValue')(
        [x_word_and_aspect, x_word])
    outp = Dense(num_class, activation="softmax", name='dense')(x)

    model = Model(inputs=[word_input, aspect_input], outputs=outp, name='model')
    adam = optimizers.adam(clipvalue=1)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    return model


def atae_lstm(max_seq_len, input_dim, output_dim, embedding_matrix, num_class=10):
    """atae_lstm"""
    aspect_input = Input(shape=(max_seq_len,), name='aspect_input')
    x_aspect = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False)(
        aspect_input)

    word_input = Input(shape=(max_seq_len,), name='word_input')
    x_word = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False)(
        word_input)

    x_word_and_aspect = concatenate([x_word, x_aspect])

    # x_word_and_aspect = SpatialDropout1D(0.2)(x_word_and_aspect)
    x_word = Bidirectional(GRU(20, return_sequences=True, dropout=0.2))(x_word_and_aspect)

    x = keras_layers.AttLayerDifferentKeyValue(20)([x_word_and_aspect, x_word])
    outp = Dense(num_class, activation="softmax")(x)

    model = Model(inputs=[word_input, aspect_input], outputs=outp)
    adam = optimizers.adam(clipvalue=1)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    return model


def rnn_attention_multi_label(max_seq_len, input_dim, output_dim, embedding_matrix, num_class):
    """rnn_attention_multi_label"""
    inp = Input(shape=(max_seq_len,))
    x = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False)(inp)
    # x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(40, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))(x)

    outputs = []
    for i in range(num_class):
        x_i = keras_layers.AttLayer(20, i)(x)
        # x_i = Dense(20, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x_i)
        # x_i = Dense(20, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x_i)
        output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01))(x_i)
        outputs.append(output)

    model = Model(inputs=inp, outputs=outputs)
    adam = optimizers.adam(clipvalue=1)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    return model


def bert_attention_multi_label(max_seq_len, input_dim, output_dim, embedding_matrix, num_class):
    """rnn_attention_multi_label"""
    config_path = data_path.bert_config_path
    checkpoint_path = data_path.bert_checkpoint_path

    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, training=False)
    bert_output = bert_model.outputs[0]

    outputs = []
    for i in range(num_class):
        x_i = keras_layers.AttLayer(20, i)(bert_output)
        x_i = Dense(20, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x_i)
        x_i = Dense(20, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x_i)
        output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01))(x_i)
        outputs.append(output)

    model = Model(inputs=bert_model.input, outputs=outputs)
    adam = optimizers.adam(clipvalue=1, lr=5e-5)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    return model


def bert_single_attention_multi_label(max_seq_len, input_dim, output_dim, embedding_matrix, num_class):
    """rnn_attention_multi_label"""
    config_path = data_path.bert_config_path
    checkpoint_path = data_path.bert_checkpoint_path

    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
    bert_output = bert_model.outputs[0]

    x_i = keras_layers.AttLayer(200, 0)(bert_output)

    outputs = []
    for i in range(num_class):
        output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01))(x_i)
        outputs.append(output)

    model = Model(inputs=bert_model.input, outputs=outputs)
    adam = optimizers.adam(clipvalue=1, lr=5e-5)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    return model


def bert_classification_hidden_multi_label(max_seq_len, input_dim, output_dim, embedding_matrix, num_class):
    """rnn_attention_multi_label"""
    config_path = data_path.bert_config_path
    checkpoint_path = data_path.bert_checkpoint_path

    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
    bert_output = bert_model.outputs[0]

    x_i = keras_layers.ClassificationHiddenLayer()(bert_output)

    outputs = []
    for i in range(num_class):
        output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01))(x_i)
        outputs.append(output)

    model = Model(inputs=bert_model.input, outputs=outputs)
    adam = optimizers.adam(clipvalue=1, lr=5e-5)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    return model


def bert_classification(max_seq_len, input_dim, output_dim, embedding_matrix, num_class):
    """rnn_attention_multi_label"""
    config_path = data_path.bert_config_path
    checkpoint_path = data_path.bert_checkpoint_path

    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
    bert_output = bert_model.outputs[0]

    x_i = keras_layers.ClassificationHiddenLayer()(bert_output)

    output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01))(x_i)

    model = Model(inputs=bert_model.input, outputs=output)
    adam = optimizers.adam(clipvalue=1, lr=5e-5)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    return model


def rnn_attention_multi_label_count(max_seq_len, input_dim, output_dim, embedding_matrix, num_class):
    """rnn_attention_multi_label"""
    inp = Input(shape=(max_seq_len,))
    x = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False)(inp)
    # x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(40, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))(x)

    outputs = []
    subject_feature = []
    loss = []
    for i in range(num_class):
        x_i = keras_layers.AttLayer(20, i)(x)
        x_i = Dense(20, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x_i)
        x_i = Dense(20, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x_i)
        subject_feature.append(x_i)
        output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01))(x_i)
        outputs.append(output)
        loss.append('binary_crossentropy')
    count_feature = concatenate(subject_feature)
    count_output = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.01))(count_feature)
    outputs.append(count_output)
    loss.append('mean_squared_error')

    model = Model(inputs=inp, outputs=outputs)
    adam = optimizers.adam(clipvalue=1)
    model.compile(loss=loss,
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    return model


def rnn_attention_multi_label_multi_input(max_seq_lens, input_dim, output_dim, embedding_matrix,
                                          num_class):
    """rnn_attention_multi_label_multi_input"""
    x_input = Input(shape=(max_seq_lens[0],))
    x = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False)(x_input)
    # x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(20, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))(x)

    y_inp = Input(shape=(max_seq_lens[1],))
    y = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False)(y_inp)
    # y = SpatialDropout1D(0.2)(y)
    y = Bidirectional(GRU(20, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))(y)

    z_inp = Input(shape=(max_seq_lens[2],))
    z = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False)(z_inp)
    # z = SpatialDropout1D(0.2)(z)
    z = Bidirectional(GRU(20, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))(z)

    conc = concatenate([x, y, z], axis=1)
    outputs = []
    for i in range(num_class):
        x_i = keras_layers.AttLayer(20, i)(conc)
        output = Dense(1, activation="sigmoid")(x_i)
        outputs.append(output)

    model = Model(inputs=[x_input, y_inp, z_inp], outputs=outputs)
    adam = optimizers.adam(clipvalue=1)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    return model


def at_lstm_with_densely_cnn(max_seq_len, input_dim, output_dim, embedding_matrix, num_class=10):
    """at_lstm_with_densely_cnn"""
    word_input = Input(shape=(max_seq_len,), name='word_input')
    x_word = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding')(word_input)

    k = 20
    conv1 = Conv1D(filters=k, kernel_size=1, activation='relu', padding='same')(x_word)

    conv2 = Conv1D(filters=k, kernel_size=2, activation='relu', padding='same')(conv1)

    conv3_1 = Conv1D(filters=k, kernel_size=2, activation='relu', padding='same')(conv1)
    conv3_2 = Conv1D(filters=k, kernel_size=2, activation='relu', padding='same')(conv2)
    conv3 = Add()([conv3_1, conv3_2])

    conv4_1 = Conv1D(filters=k, kernel_size=2, activation='relu', padding='same')(conv1)
    conv4_2 = Conv1D(filters=k, kernel_size=2, activation='relu', padding='same')(conv2)
    conv4_3 = Conv1D(filters=k, kernel_size=2, activation='relu', padding='same')(conv3)
    conv4 = Add()([conv4_1, conv4_2, conv4_3])

    attention = keras_layers.DenselyCnnAttLayer()([conv1, conv2, conv3, conv4])

    aspect_input = Input(shape=(max_seq_len,), name='aspect_input')
    x_aspect = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                         name='aspect_embedding')(aspect_input)

    x_word_and_aspect = concatenate([attention, x_aspect])

    x = keras_layers.AttLayerDifferentKeyValue(50, name='AttLayerDifferentKeyValue')(
        [x_word_and_aspect, x_word])
    outp = Dense(num_class, activation="softmax", name='dense')(x)

    model = Model(inputs=[word_input, aspect_input], outputs=outp, name='model')
    adam = optimizers.adam(clipvalue=1)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    return model

def densely_cnn(max_seq_len, input_dim, output_dim, embedding_matrix, num_class=10):
    """at_lstm_with_densely_cnn"""
    word_input = Input(shape=(max_seq_len,), name='word_input')
    x_word = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding')(word_input)

    k = 20
    conv1 = Conv1D(filters=k, kernel_size=1, activation='relu', padding='same')(x_word)

    conv2 = Conv1D(filters=k, kernel_size=2, activation='relu', padding='same')(conv1)

    conv3_1 = Conv1D(filters=k, kernel_size=2, activation='relu', padding='same')(conv1)
    conv3_2 = Conv1D(filters=k, kernel_size=2, activation='relu', padding='same')(conv2)
    conv3 = Add()([conv3_1, conv3_2])

    conv4_1 = Conv1D(filters=k, kernel_size=2, activation='relu', padding='same')(conv1)
    conv4_2 = Conv1D(filters=k, kernel_size=2, activation='relu', padding='same')(conv2)
    conv4_3 = Conv1D(filters=k, kernel_size=2, activation='relu', padding='same')(conv3)
    conv4 = Add()([conv4_1, conv4_2, conv4_3])

    attention = keras_layers.DenselyCnnAttLayer()([conv1, conv2, conv3, conv4])
    x = Flatten()(attention)
    outp = Dense(num_class, activation="softmax", name='dense')(x)

    model = Model(inputs=word_input, outputs=outp, name='model')
    adam = optimizers.adam(clipvalue=1)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    return model


def hatt_sentiment(max_sentence_num, max_seq_len, input_dim, output_dim, embedding_matrix,
                   num_class=10):
    """hatt_sentiment"""
    aspect_input = Input(shape=(1,), name='aspect_input')
    x_aspect = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False)(
        aspect_input)

    sentence_input = Input(shape=(max_seq_len,), dtype='int32')
    embedded_sequences = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding')(sentence_input)
    l_lstm = Bidirectional(GRU(20, return_sequences=True))(embedded_sequences)
    l_att = keras_layers.AttLayer(10, 0)(l_lstm)
    sentEncoder = Model(sentence_input, l_att)

    review_input = Input(shape=(max_sentence_num, max_seq_len), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    l_lstm_sent = Bidirectional(GRU(10, return_sequences=True))(review_encoder)
    l_att_sent = keras_layers.AttLayerSpecifyQuery(5)([l_lstm_sent, x_aspect])
    outp = Dense(num_class, activation="softmax", name='dense')(l_att_sent)

    model = Model(inputs=[review_input, aspect_input], outputs=outp, name='model')
    adam = optimizers.adam(clipvalue=1)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    return model


def interactive_attention_sentiment(max_seq_len, input_dim, output_dim, embedding_matrix,
                   num_class=10):
    """hatt_sentiment"""
    aspect_input = Input(shape=(1,), name='aspect_input')
    x_aspect = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False)(
        aspect_input)
    x_aspect = Bidirectional(GRU(50, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))(x_aspect)

    sentence_input = Input(shape=(max_seq_len,), dtype='int32')
    x = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding')(sentence_input)
    # x = SpatialDropout1D(0.2)(embedded_sequences)
    x = Bidirectional(GRU(50, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))(x)
    context_representation = keras_layers.AttLayerSpecifyQuery(100)([x, x_aspect])
    x_average = GlobalAveragePooling1D()(x)
    aspect_representation = keras_layers.AttLayerSpecifyQuery(100, expand_query_dims=True)([x_aspect, x_average])
    all_representation = concatenate([context_representation, aspect_representation])
    outp = Dense(num_class, activation="softmax", name='dense', kernel_regularizer=regularizers.l2(0.01))(all_representation)

    model = Model(inputs=[sentence_input, aspect_input], outputs=outp, name='model')
    adam = optimizers.adam(clipvalue=1)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    return model


def interactive_attention_sentiment_lstm(max_seq_len, input_dim, output_dim, embedding_matrix,
                   num_class=10):
    """hatt_sentiment"""
    aspect_input = Input(shape=(1,), name='aspect_input')
    x_aspect = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False)(
        aspect_input)
    x_aspect = Bidirectional(LSTM(50, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))(x_aspect)

    sentence_input = Input(shape=(max_seq_len,), dtype='int32')
    x = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding')(sentence_input)
    # x = SpatialDropout1D(0.2)(embedded_sequences)
    x = Bidirectional(LSTM(50, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))(x)
    context_representation = keras_layers.AttLayerSpecifyQuery(100)([x, x_aspect])
    x_average = GlobalAveragePooling1D()(x)
    aspect_representation = keras_layers.AttLayerSpecifyQuery(100, expand_query_dims=True)([x_aspect, x_average])
    all_representation = concatenate([context_representation, aspect_representation])
    outp = Dense(num_class, activation="softmax", name='dense', kernel_regularizer=regularizers.l2(0.01))(all_representation)

    model = Model(inputs=[sentence_input, aspect_input], outputs=outp, name='model')
    adam = optimizers.adam(clipvalue=1)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    return model


def interactive_attention_multi_label_sentiment(max_seq_len, input_dim, output_dim, embedding_matrix,
                   num_class=10):
    """hatt_sentiment"""
    aspect_input = Input(shape=(1,), name='aspect_input')
    x_aspect = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False)(
        aspect_input)
    x_aspect = Bidirectional(GRU(50, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))(x_aspect)

    sentence_input = Input(shape=(max_seq_len,), dtype='int32')
    x = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding')(sentence_input)
    # x = SpatialDropout1D(0.2)(embedded_sequences)
    x = Bidirectional(GRU(50, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))(x)
    context_representation = keras_layers.AttLayerSpecifyQuery(100)([x, x_aspect])
    x_average = GlobalAveragePooling1D()(x)
    aspect_representation = keras_layers.AttLayerSpecifyQuery(100, expand_query_dims=True)([x_aspect, x_average])
    all_representation = concatenate([context_representation, aspect_representation])
    outp = Dense(num_class, activation="sigmoid", name='dense', kernel_regularizer=regularizers.l2(0.01))(all_representation)

    model = Model(inputs=[sentence_input, aspect_input], outputs=outp, name='model')
    adam = optimizers.adam(clipvalue=1)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    return model


def aoa_sentiment(max_seq_len, input_dim, output_dim, embedding_matrix,
                   num_class=10):
    """hatt_sentiment"""
    # brnn = Bidirectional(GRU(50, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))

    aspect_input = Input(shape=(1,), name='aspect_input')
    x_aspect = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False)(
        aspect_input)
    x_aspect = Bidirectional(GRU(50, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))(x_aspect)
    # x_aspect = brnn(x_aspect)

    sentence_input = Input(shape=(max_seq_len,), dtype='int32')
    x = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding')(sentence_input)
    x = Bidirectional(GRU(50, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))(x)
    # x = brnn(x)

    # AOA
    I = Dot(axes=[2, 2])([x, x_aspect])
    context_representation = keras_layers.AOA()([x, I])

    outp = Dense(num_class, activation="softmax", name='dense', kernel_regularizer=regularizers.l2(0.001))(context_representation)

    model = Model(inputs=[sentence_input, aspect_input], outputs=outp, name='model')
    adam = optimizers.adam(clipvalue=1)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    return model


def aoa_sentiment_lstm(max_seq_len, input_dim, output_dim, embedding_matrix,
                   num_class=10):
    """hatt_sentiment"""
    # brnn = Bidirectional(GRU(50, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))

    aspect_input = Input(shape=(1,), name='aspect_input')
    x_aspect = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False)(
        aspect_input)
    x_aspect = Bidirectional(LSTM(50, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))(x_aspect)
    # x_aspect = brnn(x_aspect)

    sentence_input = Input(shape=(max_seq_len,), dtype='int32')
    x = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding')(sentence_input)
    x = Bidirectional(LSTM(50, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))(x)
    # x = brnn(x)

    # AOA
    I = Dot(axes=[2, 2])([x, x_aspect])
    context_representation = keras_layers.AOA()([x, I])

    outp = Dense(num_class, activation="softmax", name='dense', kernel_regularizer=regularizers.l2(0.001))(context_representation)

    model = Model(inputs=[sentence_input, aspect_input], outputs=outp, name='model')
    adam = optimizers.adam(clipvalue=1)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    return model


def aoa_multi_label_sentiment(max_seq_len, input_dim, output_dim, embedding_matrix,
                   num_class=10):
    """hatt_sentiment"""
    # brnn = Bidirectional(GRU(50, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))

    aspect_input = Input(shape=(1,), name='aspect_input')
    x_aspect = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False)(
        aspect_input)
    x_aspect = Bidirectional(GRU(50, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))(x_aspect)
    # x_aspect = brnn(x_aspect)

    sentence_input = Input(shape=(max_seq_len,), dtype='int32')
    x = Embedding(input_dim, output_dim, weights=[embedding_matrix], trainable=False,
                       name='word_embedding')(sentence_input)
    x = Bidirectional(GRU(50, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))(x)
    # x = brnn(x)

    # AOA
    I = Dot(axes=[2, 2])([x, x_aspect])
    context_representation = keras_layers.AOA()([x, I])

    outp = Dense(num_class, activation="sigmoid", kernel_regularizer=regularizers.l2(0.001))(context_representation)
    model = Model(inputs=[sentence_input, aspect_input], outputs=outp, name='model')
    adam = optimizers.adam(clipvalue=1)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    model.summary()
    return model
