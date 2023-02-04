from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers
import tensorflow as tf
from keras import regularizers


class AttLayer(Layer):
    def __init__(self, attention_dim, serial_num):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        self.serial_num = serial_num
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        # ait = K.print_tensor(ait, message='')
        # ait = tf.Print(ait, [ait], str(self.serial_num), summarize=40000)
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


class SpecifyKeyAttLayer(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(SpecifyKeyAttLayer, self).__init__()

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('A merge layer should be called '
                             'on a list of inputs.')
        value_input_shape = input_shape[0]
        key_input_shape = input_shape[1]
        self.W = K.variable(self.init((value_input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.W_u = K.variable(self.init((self.attention_dim, key_input_shape)))
        self.trainable_weights = [self.W, self.b, self.W_u]
        super(SpecifyKeyAttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        query = x[0]
        key = x[1]
        value = x[1]
        uit = K.tanh(K.bias_add(K.dot(key, self.W), self.b))
        u = K.dot(self.W_u, query)
        ait = K.dot(uit, u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        # ait = K.print_tensor(ait, message='')
        # ait = tf.Print(ait, [ait], "", summarize=40000)
        ait = K.expand_dims(ait)

        weighted_input = value * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        value_input_shape = input_shape[1]
        return (value_input_shape[0], value_input_shape[-1])


class AttLayerDifferentKeyValue(Layer):
    def __init__(self, attention_dim, **kwargs):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayerDifferentKeyValue, self).__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('A merge layer should be called '
                             'on a list of inputs.')
        key_input_shape = input_shape[0]
        self.W = K.variable(self.init((key_input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayerDifferentKeyValue, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        key = x[0]
        value = x[1]
        uit = K.tanh(K.bias_add(K.dot(key, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        # ait = K.print_tensor(ait, message='')
        # ait = tf.Print(ait, [ait], "", summarize=40000)
        ait = K.expand_dims(ait)

        weighted_input = value * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        value_input_shape = input_shape[1]
        return (value_input_shape[0], value_input_shape[-1])


class AttLayerDifferentKeyValue(Layer):
    def __init__(self, attention_dim, kernel_regularizer=None, bias_regularizer=None, **kwargs):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        super(AttLayerDifferentKeyValue, self).__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('A merge layer should be called '
                             'on a list of inputs.')
        key_input_shape = input_shape[0]
        # self.W = K.variable(self.init((key_input_shape[-1], self.attention_dim)))
        # self.b = K.variable(self.init((self.attention_dim, )))
        # self.u = K.variable(self.init((self.attention_dim, 1)))
        self.W = self.add_weight(name='kernel', shape=(key_input_shape[-1], self.attention_dim),
                                 initializer='uniform', trainable=True, regularizer=self.kernel_regularizer)
        self.u = self.add_weight(name='kernel_u', shape=(self.attention_dim, 1),
                                 initializer='uniform', trainable=True, regularizer=self.kernel_regularizer)
        self.b = self.add_weight(name='bias', shape=(self.attention_dim, ),
                                 initializer='uniform', trainable=True,
                                 regularizer=self.bias_regularizer)
        # self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayerDifferentKeyValue, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        key = x[0]
        value = x[1]
        uit = K.tanh(K.bias_add(K.dot(key, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        # ait = K.print_tensor(ait, message='')
        # ait = tf.Print(ait, [ait], "", summarize=40000)
        ait = K.expand_dims(ait)

        weighted_input = value * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        value_input_shape = input_shape[1]
        return (value_input_shape[0], value_input_shape[-1])


class AttLayerSpecifyQuery(Layer):
    def __init__(self, attention_dim, expand_query_dims=False, **kwargs):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.expand_query_dims = expand_query_dims
        self.attention_dim = attention_dim
        super(AttLayerSpecifyQuery, self).__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('A merge layer should be called '
                             'on a list of inputs.')
        key_value_input_shape = input_shape[0]
        query_input_shape = input_shape[1]
        self.seq_len = key_value_input_shape[1]
        self.W = K.variable(self.init((key_value_input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        # self.u_w = K.variable(self.init((query_input_shape[-1], self.attention_dim)))
        self.trainable_weights = [self.W, self.b]
        super(AttLayerSpecifyQuery, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        key = x[0]
        value = x[0]
        query = x[1]
        if self.expand_query_dims:
            query = K.expand_dims(query, axis=1)
        uit = K.tanh(K.bias_add(K.dot(key, self.W), self.b))
        # u = K.dot(query, self.u_w)
        u = query
        u_repeat = K.repeat_elements(u, self.seq_len, 1)
        ait = uit * u_repeat
        ait = K.sum(ait, axis=-1, keepdims=False)
        ait = K.softmax(ait)
        ait = K.expand_dims(ait)
        # ait = K.print_tensor(ait, message='')
        # ait = tf.Print(ait, [ait], "", summarize=40000)

        weighted_input = value * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        value_input_shape = input_shape[0]
        return (value_input_shape[0], value_input_shape[-1])


class AOA(Layer):
    def __init__(self, **kwargs):
        super(AOA, self).__init__(**kwargs)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        sentence = x[0]
        I = x[1]
        # I = tf.Print(I, [I], "I", summarize=40000)

        alpha = K.softmax(I, axis=1)
        # alpha = tf.Print(alpha, [alpha], "alpha", summarize=40000)
        beta = K.softmax(I, axis=2)
        # beta = tf.Print(beta, [beta], "beta", summarize=40000)

        beta_bar = K.mean(beta, axis=1, keepdims=True)
        # beta_bar = tf.Print(beta_bar, [beta_bar], "beta_bar", summarize=40000)

        gamma = K.batch_dot(alpha, beta_bar, axes=[2, 2])
        # gamma = tf.Print(gamma, [gamma], "gamma", summarize=40000)
        weighted_input = sentence * gamma
        output = K.sum(weighted_input, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        value_input_shape = input_shape[0]
        return (value_input_shape[0], value_input_shape[-1])


class SumTensor(Layer):
    def __init__(self, **kwargs):
        super(SumTensor, self).__init__(**kwargs)

    def call(self, x, mask=None):
        output = K.sum(x, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


class DenselyCnnAttLayer(Layer):
    def __init__(self):
        self.init = initializers.get('normal')
        self.supports_masking = True
        super(DenselyCnnAttLayer, self).__init__()

    def build(self, input_shape):
        densely_layer_num = len(input_shape)
        self.seq_len = input_shape[0][1]
        self.Ws = []
        for i in range(self.seq_len):
            W = K.variable(self.init((densely_layer_num, densely_layer_num)))
            self.Ws.append(W)
        self.trainable_weights = self.Ws
        super(DenselyCnnAttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # 取得所有x_i的向量
        position_matrix = []
        for i in range(self.seq_len):
            position_i_vectors = []
            for j in range(len(x)):
                position_i_vectors.append(x[j][:, i, :])
            position_i_vector_stack = K.stack(position_i_vectors, axis=1)
            position_i_sum = K.sum(position_i_vector_stack, axis=-1)
            a_i = K.dot(position_i_sum, self.Ws[i])
            # a_i = K.print_tensor(a_i, message='a_i')
            a_i = K.softmax(a_i)
            # a_i = K.print_tensor(a_i, message='softmax')
            a_i = K.expand_dims(a_i)
            x_i_atten = position_i_vector_stack * a_i
            x_i_atten = K.sum(x_i_atten, axis=1)
            position_matrix.append(x_i_atten)
        position_matrix_stack =K.stack(position_matrix, 1)

        return position_matrix_stack

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], input_shape[0][2]


class ClassificationHiddenLayer(Layer):
    def __init__(self):
        super(ClassificationHiddenLayer, self).__init__()

    def build(self, input_shape):
        super(ClassificationHiddenLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # 取得所有x_i的向量

        result = x[:, 0, :]
        return result

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]


class Position_Embedding(Layer):

    def __init__(self, size=None, mode='sum', **kwargs):
        self.size = size  # 必须为偶数
        self.mode = mode
        super(Position_Embedding, self).__init__(**kwargs)

    def call(self, x):
        if (self.size == None) or (self.mode == 'sum'):
            self.size = int(x.shape[-1])
        batch_size, seq_len = K.shape(x)[0], K.shape(x)[1]
        position_j = 1. / K.pow(10000., \
                                2 * K.arange(self.size / 2, dtype='float32' \
                                             ) / self.size)
        position_j = K.expand_dims(position_j, 0)
        position_i = K.cumsum(K.ones_like(x[:, :, 0]), 1) - 1  # K.arange不支持变长，只好用这种方法生成
        position_i = K.expand_dims(position_i, 2)
        position_ij = K.dot(position_i, position_j)
        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)
        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            return K.concatenate([position_ij, x], 2)

    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2] + self.size)
