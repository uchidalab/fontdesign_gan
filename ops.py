import tensorflow as tf
import numpy as np

"""
Operands for Models
"""


def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)


def batch_norm(x, is_train, eps=1e-5, decay=0.9, name='batch_norm'):
    return tf.contrib.layers.batch_norm(x, decay=decay, updates_collections=None, epsilon=eps,
                                        scale=True, is_training=is_train, scope=name)


def layer_norm(x, name='layer_norm'):
    return tf.contrib.layers.layer_norm(x, scope=name)


def linear(x, n_out, name='linear'):
    with tf.variable_scope(name):
        shape = x.shape.as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        x = tf.reshape(x, [-1, dim])

        w_init = tf.truncated_normal_initializer(0.0, np.sqrt(1.0 / n_out))
        w = tf.get_variable('w', [x.shape[-1], n_out], initializer=w_init)
        b_init = tf.constant_initializer(0.0)
        b = tf.get_variable('b', [n_out], initializer=b_init)

        return tf.matmul(x, w) + b


def conv2d(x, n_out, k, s, p, stddev=0.02, name='conv2d'):
    with tf.variable_scope(name):
        n_in = x.shape[-1]
        strides = [1, s, s, 1]

        w_init = tf.truncated_normal_initializer(stddev=stddev)
        w = tf.get_variable('w', [k, k, n_in, n_out], initializer=w_init)
        conv = tf.nn.conv2d(x, w, strides=strides, padding=p)

        b_init = tf.constant_initializer(0.0)
        b = tf.get_variable('b', shape=(n_out,), initializer=b_init)
        conv = tf.reshape(tf.nn.bias_add(conv, b), conv.get_shape())

        return conv


def deconv2d(x, out_shape, k, s, p, stddev=0.02, name='deconv2d'):
    with tf.variable_scope(name):
        inp_shape = x.get_shape().as_list()
        strides = [1, s, s, 1]

        w_init = tf.random_normal_initializer(stddev=stddev)
        w = tf.get_variable('w', [k, k, out_shape[-1], inp_shape[-1]], initializer=w_init)
        deconv = tf.nn.conv2d_transpose(x, w, output_shape=out_shape, strides=strides, padding=p)

        b_init = tf.constant_initializer(0.0)
        b = tf.get_variable('b', shape=(out_shape[-1],), initializer=b_init)
        deconv = tf.reshape(tf.nn.bias_add(deconv, b), deconv.get_shape())

        return deconv


def upsample2x(x):
    return tf.depth_to_space(tf.concat([x for _ in range(4)], axis=3), 2)


def downsample2x(x):
    return tf.add_n([x[:, ::2, ::2, :], x[:, 1::2, ::2, :], x[:, ::2, 1::2, :], x[:, 1::2, 1::2, :]]) / 4.


def maxpool2d(x, k, s, p, name='maxpool2d'):
    strides = [1, s, s, 1]
    ksize = [1, k, k, 1]

    return tf.nn.max_pool(x, ksize=ksize, strides=strides, padding=p, name=name)


def fc(x, n_out, name='fc'):

    with tf.variable_scope(name):

        shape = x.shape.as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        x = tf.reshape(x, [-1, dim])

        w_init = tf.truncated_normal_initializer(0.0, np.sqrt(1.0 / n_out))
        w = tf.get_variable('w', [x.shape[-1], n_out], initializer=w_init)

        b_init = tf.constant_initializer(0.0)
        b = tf.get_variable('b', [n_out], initializer=b_init)

        return tf.matmul(x, w) + b


def average_gradients(tower_grads):
    """
    This code is borrowed from this:
    https://github.com/tensorflow/tensorflow/blob/r0.7/tensorflow/models/image/cifar10/cifar10_multi_gpu_train.py
    """
    averaged_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, axis=0)
            grads.append(expanded_g)
        grad = tf.concat(grads, axis=0)
        grad = tf.reduce_mean(grad, axis=0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        averaged_grads.append(grad_and_var)
    return averaged_grads
