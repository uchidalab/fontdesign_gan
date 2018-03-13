import tensorflow as tf

from ops import lrelu, batch_norm, linear, conv2d, maxpool2d, fc, layer_norm, upsample2x, downsample2x

"""
2 type models are defined:
- DCGAN [Radford+, ICLR2016]
- ResNet [He+, CVPR2016]
  - This model needs a lot of GPU memory, so you should reduce batch size.
"""


class GeneratorDCGAN():

    def __init__(self, img_size=(128, 128), img_dim=1, z_size=100, k_size=5, layer_n=3,
                 smallest_hidden_unit_n=128, is_bn=True):
        self.img_size = img_size
        self.img_dim = img_dim
        self.z_size = z_size
        self.k_size = k_size
        self.layer_n = layer_n
        self.smallest_hidden_unit_n = smallest_hidden_unit_n
        self.is_bn = is_bn

    def __call__(self, x, is_reuse=False, is_train=True):
        with tf.variable_scope('generator') as scope:
            if is_reuse:
                scope.reuse_variables()

            unit_size = self.img_size[0] // (2 ** self.layer_n)
            unit_n = self.smallest_hidden_unit_n * (2 ** (self.layer_n - 1))
            batch_size = int(x.shape[0])

            with tf.variable_scope('pre'):
                x = linear(x, unit_size * unit_size * unit_n)
                x = tf.reshape(x, (batch_size, unit_size, unit_size, unit_n))
                if self.is_bn:
                    x = batch_norm(x, is_train)
                x = tf.nn.relu(x)

            for i in range(self.layer_n):
                with tf.variable_scope('layer{}'.format(i)):
                    if i == self.layer_n - 1:
                        unit_n = self.img_dim
                    else:
                        unit_n = self.smallest_hidden_unit_n * (2 ** (self.layer_n - i - 2))
                    x_shape = x.get_shape().as_list()
                    x = tf.image.resize_bilinear(x, (x_shape[1] * 2, x_shape[2] * 2))
                    x = conv2d(x, unit_n, self.k_size, 1, 'SAME')
                    if i != self.layer_n - 1:
                        if self.is_bn:
                            x = batch_norm(x, is_train)
                        x = tf.nn.relu(x)
            x = tf.nn.tanh(x)

            return x


class DiscriminatorDCGAN():

    def __init__(self, img_size=(128, 128), img_dim=1, k_size=5, layer_n=3, smallest_hidden_unit_n=128, is_bn=True):
        self.img_size = img_size
        self.img_dim = img_dim
        self.k_size = k_size
        self.layer_n = layer_n
        self.smallest_hidden_unit_n = smallest_hidden_unit_n
        self.is_bn = is_bn

    def __call__(self, x, is_reuse=False, is_train=True):
        with tf.variable_scope('discriminator') as scope:
            if is_reuse:
                scope.reuse_variables()

            unit_n = self.smallest_hidden_unit_n
            batch_size = int(x.shape[0])

            for i in range(self.layer_n):
                with tf.variable_scope('layer{}'.format(i + 1)):
                    x = conv2d(x, unit_n, self.k_size, 2, 'SAME')
                    if self.is_bn and i != 0:
                        x = batch_norm(x, is_train)
                    x = lrelu(x)
                    unit_n = self.smallest_hidden_unit_n * (2 ** (i + 1))

            x = tf.reshape(x, (batch_size, -1))
            x = linear(x, 1)

            return x


class GeneratorResNet():

    def __init__(self, k_size=3, smallest_unit_n=64):
        self.k_size = k_size
        self.smallest_unit_n = smallest_unit_n

    def _residual_block(self, x, n_out, is_train, name='residual'):
        with tf.variable_scope(name):
            with tf.variable_scope('shortcut'):
                x1 = upsample2x(x)
                x1 = conv2d(x1, n_out, self.k_size, 1, 'SAME')
            with tf.variable_scope('normal'):
                x2 = batch_norm(x, is_train, name='batch_norm_0')
                x2 = tf.nn.relu(x2)
                x2 = upsample2x(x2)
                x2 = conv2d(x2, n_out, self.k_size, 1, 'SAME', name='conv2d_0')
                x2 = batch_norm(x2, is_train, name='batch_norm_1')
                x2 = tf.nn.relu(x2)
                x2 = conv2d(x2, n_out, self.k_size, 1, 'SAME', name='conv2d_1')
            return x1 + x2

    def __call__(self, x, is_train=True, is_reuse=False):
        with tf.variable_scope('generator') as scope:
            if is_reuse:
                scope.reuse_variables()

            with tf.variable_scope('first'):
                x = linear(x, 4 * 4 * 8 * self.smallest_unit_n)
            x = tf.reshape(x, [-1, 4, 4, 8 * self.smallest_unit_n])

            for i, times in enumerate([8, 4, 2, 1]):
                x = self._residual_block(x, times * self.smallest_unit_n, is_train, 'residual_{}'.format(i))

            with tf.variable_scope('last'):
                x = batch_norm(x, is_train)
                x = tf.nn.relu(x)
                x = conv2d(x, 3, self.k_size, 1, 'SAME')

            x = tf.tanh(x)

            return x


class DiscriminatorResNet():

    def __init__(self, k_size=3, smallest_unit_n=64):
        self.k_size = k_size
        self.smallest_unit_n = smallest_unit_n

    def _residual_block(self, x, n_out, name='residual'):
        with tf.variable_scope(name):
            with tf.variable_scope('shortcut'):
                x1 = downsample2x(x)
                x1 = conv2d(x1, n_out, self.k_size, 1, 'SAME')
            with tf.variable_scope('normal'):
                x2 = layer_norm(x, name='layer_norm_0')
                x2 = tf.nn.relu(x2)
                x2 = conv2d(x2, n_out, self.k_size, 1, 'SAME', name='conv2d_0')
                x2 = layer_norm(x2, name='layer_norm_1')
                x2 = tf.nn.relu(x2)
                x2 = downsample2x(x2)
                x2 = conv2d(x2, n_out, self.k_size, 1, 'SAME', name='conv2d_1')
            return x1 + x2

    def __call__(self, x, is_train=True, is_reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if is_reuse:
                scope.reuse_variables()

            with tf.variable_scope('first'):
                x = conv2d(x, self.smallest_unit_n, self.k_size, 1, 'SAME')

            for i, times in enumerate([2, 4, 8, 8]):
                x = self._residual_block(x, times * self.smallest_unit_n, 'residual_{}'.format(i))

            x = tf.reshape(x, [-1, 4 * 4 * 8 * self.smallest_unit_n])

            with tf.variable_scope('last'):
                x = linear(x, 1)

            return x


class Classifier():

    def __init__(self, img_size, img_dim, k_size, class_n, smallest_unit_n=64):
        self.img_size = img_size
        self.img_dim = img_dim
        self.k_size = k_size
        self.class_n = class_n
        self.smallest_unit_n = smallest_unit_n

    def __call__(self, x, is_reuse=False, is_train=True):
        with tf.variable_scope('classifier') as scope:

            if is_reuse:
                scope.reuse_variables()

            unit_n = self.smallest_unit_n
            conv_ns = [2, 2]

            for layer_i, conv_n in enumerate(conv_ns):
                with tf.variable_scope('layer{}'.format(layer_i)):
                    for conv_i in range(conv_n):
                        x = conv2d(x, unit_n, self.k_size, 1, 'SAME', name='conv2d_{}'.format(conv_i))
                        x = tf.nn.relu(x)
                    x = maxpool2d(x, self.k_size, 2, 'SAME')
                unit_n *= 2

            unit_n = 256
            fc_n = 1

            for layer_i in range(len(conv_ns), len(conv_ns) + fc_n):
                with tf.variable_scope('layer{}'.format(layer_i)):
                    x = fc(x, unit_n)
                    x = tf.nn.relu(x)
                    x = batch_norm(x, is_train)
                    x = tf.nn.dropout(x, 0.5)

            with tf.variable_scope('output'.format(layer_i)):
                x = fc(x, self.class_n)

            return x
