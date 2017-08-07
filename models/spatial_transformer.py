# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import tensorflow as tf


def localisation_net(input_tensor, eval=False, name='localisation_net'):

    CONV1_DEPTH = 32
    CONV2_DEPTH = 64
    FC1_SIZE = 128
    OUTPUT_SIZE = 3

    batch_size = input_tensor.get_shape().as_list()[0]

    with tf.variable_scope(name):
        with tf.variable_scope('conv1'):
            conv1 = tf.layers.conv2d(merge,
                                     filters=CONV1_DEPTH,
                                     kernel_size=(5, 5),
                                     strides=(1, 1),
                                     padding='valid',
                                     activation=tf.nn.relu,
                                     use_bias=True,
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                     bias_initializer=tf.constant_initializer(0.1))
        with tf.variable_scope('pool1'):
            pool1 = tf.layers.average_pooling2d(conv1,
                                                pool_size=(2, 2),
                                                strides=(2, 2),
                                                padding='valid')
        with tf.variable_scope('conv2'):
            conv2 = tf.layers.conv2d(pool1,
                                     filters=CONV2_DEPTH,
                                     kernel_size=(3, 3),
                                     strides=(1, 1),
                                     padding='valid',
                                     activation=tf.nn.relu,
                                     use_bias=True,
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                     bias_initializer=tf.constant_initializer(0.1))
        with tf.variable_scope('pool2'):
            pool2 = tf.layers.average_pooling2d(conv2,
                                                pool_size=(2, 2),
                                                strides=(2, 2),
                                                padding='valid')
        with tf.variable_scope('fc1'):
            flatten = tf.reshape(pool2, shape=(batch_size, -1))
            fc1 = tf.layers.dense(flatten,
                                  units=FC1_SIZE,
                                  activation=tf.nn.relu,
                                  use_bias=True,
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                  bias_initializer=tf.constant_initializer(0.1),
                                  kernel_regularizer=tf.nn.l2_loss)
        with tf.variable_scope('output'):
            fc2 = tf.layers.dense(fc1,
                                  units=OUTPUT_SIZE,
                                  activation=tf.nn.relu,
                                  use_bias=True,
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                  bias_initializer=tf.constant_initializer(0.1),
                                  kernel_regularizer=tf.nn.l2_loss)
    return fc2



def affine_transformer(U, theta, out_size, name='SpatialTransformer', **kwargs):
    """Spatial Transformer Layer

    Implements a spatial transformer layer as described in [1]_.
    Based on [2]_ and edited by David Dao for Tensorflow.

    Parameters
    ----------
    U : float
        The output of a convolutional net should have the
        shape [num_batch, height, width, num_channels].
    theta: float
        The output of the
        localisation network should be [num_batch, 6].
    out_size: tuple of two ints
        The size of the output of the network (height, width)

    References
    ----------
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015
    .. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py

    Notes
    -----
    To initialize the network to the identity transform init
    ``theta`` to :
        identity = np.array([[1., 0., 0.],
                             [0., 1., 0.]])
        identity = identity.flatten()
        theta = tf.Variable(initial_value=identity)

    """

    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.transpose(
                tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
            rep = tf.cast(rep, 'int32')
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

    def _interpolate(im, x, y, out_size):
        with tf.variable_scope('_interpolate'):
            # constants
            num_batch = tf.shape(im)[0]
            height = tf.shape(im)[1]
            width = tf.shape(im)[2]
            channels = tf.shape(im)[3]

            x = tf.cast(x, 'float32')
            y = tf.cast(y, 'float32')
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            zero = tf.zeros([], dtype='int32')
            max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
            max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

            # scale indices from [-1, 1] to [0, width/height]
            x = (x + 1.0)*(width_f) / 2.0
            y = (y + 1.0)*(height_f) / 2.0

            # do sampling
            x0 = tf.cast(tf.floor(x), 'int32')
            x1 = x0 + 1
            y0 = tf.cast(tf.floor(y), 'int32')
            y1 = y0 + 1

            x0 = tf.clip_by_value(x0, zero, max_x)
            x1 = tf.clip_by_value(x1, zero, max_x)
            y0 = tf.clip_by_value(y0, zero, max_y)
            y1 = tf.clip_by_value(y1, zero, max_y)
            dim2 = width
            dim1 = width*height
            base = _repeat(tf.range(num_batch)*dim1, out_height*out_width)  # [batch_num, oh * ow]
            base_y0 = base + y0*dim2
            base_y1 = base + y1*dim2
            idx_a = base_y0 + x0
            idx_b = base_y1 + x0
            idx_c = base_y0 + x1
            idx_d = base_y1 + x1

            print 'idx_a shape', idx_a.get_shape().as_list()

            # use indices to lookup pixels in the flat image and restore
            # channels dim
            im_flat = tf.reshape(im, tf.stack([-1, channels]))
            im_flat = tf.cast(im_flat, 'float32')
            Ia = tf.gather(im_flat, idx_a)
            Ib = tf.gather(im_flat, idx_b)
            Ic = tf.gather(im_flat, idx_c)
            Id = tf.gather(im_flat, idx_d)

            # and finally calculate interpolated values
            x0_f = tf.cast(x0, 'float32')
            x1_f = tf.cast(x1, 'float32')
            y0_f = tf.cast(y0, 'float32')
            y1_f = tf.cast(y1, 'float32')
            wa = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
            wb = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
            wc = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
            wd = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)
            output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
            return output

    def _meshgrid(height, width):
        with tf.variable_scope('_meshgrid'):
            # This should be equivalent to:
            #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
            #                         np.linspace(-1, 1, height))
            #  ones = np.ones(np.prod(x_t.shape))
            #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
            x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                            tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
            y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                            tf.ones(shape=tf.stack([1, width])))

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))

            ones = tf.ones_like(x_t_flat)
            grid = tf.concat(axis=0, values=[x_t_flat, y_t_flat, ones])
            return grid

    def _transform(theta, input_dim, out_size):
        with tf.variable_scope('_transform'):
            input_shape = input_dim.get_shape().as_list()
            num_batch = input_shape[0]
            height = input_shape[1]
            width = input_shape[2]
            num_channels = input_shape[3]
            theta = tf.reshape(theta, (-1, 2, 3))
            theta = tf.cast(theta, 'float32')

            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            grid = _meshgrid(out_height, out_width)
            grid = tf.reshape(grid, [-1])
            grid = tf.tile(grid, tf.stack([num_batch]))
            grid = tf.reshape(grid, tf.stack([num_batch, 3, -1]))


            # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
            T_g = tf.matmul(theta, grid)
            x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
            y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
            x_s_flat = tf.reshape(x_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])

            input_transformed = _interpolate(
                input_dim, x_s_flat, y_s_flat,
                out_size)

            output = tf.reshape(
                input_transformed, tf.stack([num_batch, out_height, out_width, num_channels]))
            return output

    with tf.variable_scope(name):
        output = _transform(theta, U, out_size)
        return output


def rotate_and_translation_transformer(U, theta, out_size, name='rotate_and_translation_transformer'):
    def _interpolate(im, x, y, out_size):
        with tf.variable_scope('_interpolate'):
            # constants
            im_shape = im.get_shape().as_list()
            num_batch = im_shape[0]
            height = im_shape[1]
            width = im_shape[2]
            channels = im_shape[3]

            x = tf.cast(x, dtype=tf.float32)
            y = tf.cast(y, dtype=tf.float32)
            height_f = tf.cast(height, dtype=tf.float32)
            width_f = tf.cast(width, dtype=tf.float32)
            out_height = out_size[0]
            out_width = out_size[1]
            zero = tf.zeros([], dtype=tf.int32)
            max_y = tf.cast(tf.shape(im)[1] - 1, dtype=tf.int32)
            max_x = tf.cast(tf.shape(im)[2] - 1, dtype=tf.int32)

            # scale indices from [-1, 1] to [0, width/height]
            # nearest
            _x = tf.cast((x + 1.0)*(width_f) / 2.0 + 0.5, dtype=tf.int32)
            _y = tf.cast((y + 1.0)*(height_f) / 2.0 + 0.5, dtype=tf.int32)
            _x = tf.clip_by_value(_x, zero, max_x)
            _y = tf.clip_by_value(_y, zero, max_y)

            batch_range = tf.range(num_batch) * width * height
            base_batch = tf.reshape(tf.stack([batch_range for _ in range(out_height*out_width)], axis=1), [-1])
            base_y = base_batch + _y * width
            idx = base_y + _x
            im_flat = tf.reshape(im, tf.stack([-1, channels]))
            output = tf.gather(im_flat, idx)
            return output
        
    def _meshgrid(height, width):
        with tf.variable_scope('_meshgrid'):
            # This should be equivalent to:
            #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
            #                         np.linspace(-1, 1, height))
            #  ones = np.ones(np.prod(x_t.shape))
            #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])

            # shape of x_t, y_t is [height, width]
            x_t = tf.matmul(tf.ones(shape=[height, 1]),
                            tf.expand_dims(tf.linspace(-1.0, 1.0, width), 0))
            y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                            tf.ones(shape=[1, width]))

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))

            ones = tf.ones_like(x_t_flat)
            grid = tf.concat(axis=0, values=[x_t_flat, y_t_flat, ones])
            return grid

    def _transform(theta, input_dim, out_size):
        with tf.variable_scope('_transform'):
            input_shape = input_dim.get_shape().as_list()
            num_batch = input_shape[0]
            num_channels = input_shape[3]

            # theta is [Tx, Ty, a], a is the rotate angle
            # A_theta = [
            # cos_a    -sin_a    Tx
            # sin_a    cos_a     Ty
            # ]
            theta = tf.cast(theta, dtype=tf.float32)
            angle = theta[:,2]
            sin_a = tf.sin(angle)
            cos_a = tf.cos(angle)
            A_theta = tf.stack([cos_a, -sin_a, theta[:,0], sin_a, cos_a, theta[:,1]], axis=1)
            A_theta = tf.reshape(A_theta, shape=[num_batch, 2, 3])

            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            out_height = out_size[0]
            out_width = out_size[1]
            grid = _meshgrid(out_height, out_width)
            grid = tf.reshape(grid, [-1])
            grid = tf.tile(grid, tf.stack([num_batch]))
            grid = tf.reshape(grid, [num_batch, 3, -1])


            # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
            T_g = tf.matmul(A_theta, grid)
            x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
            y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
            x_s_flat = tf.reshape(x_s, [ -1])
            y_s_flat = tf.reshape(y_s, [ -1])

            input_transformed = _interpolate(
                input_dim, x_s_flat, y_s_flat,
                out_size)

            output = tf.reshape(
                input_transformed, [num_batch, out_height, out_width, num_channels])
            return output

    with tf.variable_scope(name):
        output = _transform(theta, U, out_size)
        return output


if __name__ == '__main__':
    U = tf.ones(shape=[128,200,200,3], dtype=tf.float32)
    theta = tf.ones(shape=[128,3], dtype=tf.float32)
    rotate_and_translation_transformer(U, theta, (150,150))
    # s = tf.reshape(tf.stack([tf.range(5) for _ in range(3)], axis=1), [-1])
    # with tf.Session() as sess:
    #     print sess.run(s)