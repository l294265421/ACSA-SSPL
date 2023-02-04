# -*- coding: utf-8 -*-
"""

Very Deep Convolutional Networks for Text Classification

Date:    2018/9/28 15:14
"""

from keras.engine import Layer
from keras.engine import InputSpec
from keras.layers import Flatten
import tensorflow as tf


class KMaxPooling(Layer):
    """
    K-max pooling layer that extracts the k-highest activations from a sequence (2nd dimension).
    TensorFlow backend.
    """
    def __init__(self, k=1, sorted=True, **kwargs):
        super(KMaxPooling, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k = k
        self.sorted = sorted

    def compute_output_shape(self, input_shape):
        """compute_output_shape"""
        return (input_shape[0], self.k, input_shape[2])

    def call(self, inputs):
        """call"""
        # swap last two dimensions since top_k will be applied along the last dimension
        shifted_inputs = tf.transpose(inputs, [0, 2, 1])
        
        # extract top_k, returns two tensors [values, indices]
        top_k = tf.nn.top_k(shifted_inputs, k=self.k, sorted=self.sorted)[0]
        
        # return flattened output
        return tf.transpose(top_k, [0, 2, 1])