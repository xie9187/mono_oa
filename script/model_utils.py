import numpy as np
import tensorflow as tf


def Conv2D(inputs,
           num_outputs,
           kernel_size,
           strides,
           scope=None,
           activation=tf.nn.leaky_relu,
           trainable=True,
           reuse=False):
    outputs = tf.contrib.layers.conv2d(inputs=inputs,
                                       num_outputs=num_outputs,
                                       kernel_size=kernel_size,
                                       stride=strides,
                                       padding='SAME',
                                       activation_fn=activation,
                                       trainable=trainable,
                                       reuse=reuse,
                                       scope=scope or 'conv2d')
    return outputs


def _lstm_cell(n_hidden, n_layers, name=None, reuse=False):
    """select proper lstm cell."""
    cell = tf.contrib.rnn.LSTMCell(n_hidden, state_is_tuple=True, reuse=reuse, name=name or 'lstm_cell')
    if n_layers > 1:
        cell = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.LSTMCell(
             n_hidden, state_is_tuple=True, reuse=reuse, name=name or 'lstm_cell') for _ in range(n_layers)])
    return cell



