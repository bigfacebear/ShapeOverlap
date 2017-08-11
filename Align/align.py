"""
st5.py

This file contains the graph structure for simpletrain5.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

import tensorflow as tf
import numpy as np

import align_FLAGS as FLAGS
import align_input


# Global constants describing the MSHAPES data set.
IMAGE_SIZE = FLAGS.IMAGE_SIZE
NUM_CLASSES = FLAGS.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
TOWER_NAME = FLAGS.TOWER_NAME

# Constants describing the training process.
MOVING_AVERAGE_DECAY = FLAGS.MOVING_AVERAGE_DECAY  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = FLAGS.NUM_EPOCHS_PER_DECAY  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = FLAGS.LEARNING_RATE_DECAY_FACTOR  # Learning rate decay factor.
INITIAL_LEARNING_RATE = FLAGS.INITIAL_LEARNING_RATE  # Initial learning rate.

def rotate_and_translation_transformer(U, theta, out_size, name='rotate_and_translation_transformer'):
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
            angle = theta[:, 2]
            sin_a = tf.sin(angle)
            cos_a = tf.cos(angle)
            A_theta = tf.stack([cos_a, -sin_a, theta[:, 0], sin_a, cos_a, theta[:, 1]], axis=1)
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
            x_s_flat = tf.reshape(x_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])

            input_transformed = _interpolate(
                input_dim, x_s_flat, y_s_flat,
                out_size)

            output = tf.reshape(
                input_transformed, [num_batch, out_height, out_width, num_channels])
            return output

    with tf.variable_scope(name):
        output = _transform(theta, U, out_size)
        return output



def localisation_net(input_tensor, eval=False, name='localisation_net'):
    CONV1_DEPTH = 16
    CONV2_DEPTH = 16

    FC1_NUM = 786
    FC2_NUM = 384

    theta_param_num = 3

    channel_num = input_tensor.get_shape().as_list()[3]
    with tf.variable_scope(name):
        # conv1
        with tf.variable_scope('conv1') as scope:
            kernel = _variable_with_weight_decay('weights',
                                                 shape=[5, 5, channel_num, CONV1_DEPTH],
                                                 stddev=5e-3,
                                                 wd=0.0)
            conv = tf.nn.conv2d(input_tensor, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu('biases', [CONV1_DEPTH], tf.constant_initializer(1e-2))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(pre_activation, name=scope.name)

            # pool1
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                   padding='SAME', name='pool')
            # norm1
            norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                              name='norm1')

        # conv2
        with tf.variable_scope('conv2') as scope:
            kernel = _variable_with_weight_decay('weights',
                                                 shape=[5, 5, CONV1_DEPTH, CONV2_DEPTH],
                                                 stddev=5e-2,
                                                 wd=0.0)
            conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu('biases', [CONV2_DEPTH], tf.constant_initializer(0.1))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(pre_activation, name=scope.name)

            # norm2
            norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                              name='norm2')
            # pool2
            pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                                   strides=[1, 2, 2, 1], padding='SAME', name='pool1')
        # FC1
        with tf.variable_scope('FC1') as scope:
            # Move everything into depth so we can perform a single matrix multiply.
            reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
            dim = reshape.get_shape()[1].value
            weights = _variable_with_weight_decay('weights', shape=[dim, FC1_NUM],
                                                  stddev=0.04, wd=0.004)
            biases = _variable_on_cpu('biases', [FC1_NUM], tf.constant_initializer(0.1))
            fc1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
            keep_prob = FLAGS.KEEP_PROB if eval else 1.0
            fc1_dropout = tf.nn.dropout(fc1, keep_prob=keep_prob)

        # FC2
        with tf.variable_scope('FC2') as scope:
            weights = _variable_with_weight_decay('weights', shape=[FC1_NUM, FC2_NUM],  # 192
                                                  stddev=0.04, wd=0.004)
            biases = _variable_on_cpu('biases', [FC2_NUM], tf.constant_initializer(0.1))
            fc2 = tf.nn.relu(tf.matmul(fc1_dropout, weights) + biases, name=scope.name)

        # linear layer(WX + b),
        # and performs the softmax internally for efficiency.
        with tf.variable_scope('theta') as scope:
            weights = _variable_with_weight_decay('weights', [FC2_NUM, theta_param_num],
                                                  stddev=1 / float(FC2_NUM), wd=0.0)
            biases = _variable_on_cpu('biases', [theta_param_num],
                                      tf.constant_initializer(0.0))
            theta = tf.add(tf.matmul(fc2, weights), biases, name=scope.name)

    return theta


def spatial_transformer_layer(input_tensor, eval=False, name='spatial_transformer'):
    with tf.variable_scope(name):
        shape = input_tensor.get_shape().as_list()
        height, width = shape[1], shape[2]
        theta = localisation_net(input_tensor, eval=eval)
        return rotate_and_translation_transformer(input_tensor, theta, (height // 2, width // 2))


def inputs(eval_data):
    """Construct input for MSHAPES evaluation using the Reader ops.
    Args:
      eval_data: bool, indicating if one should use the train or eval data set.
    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 6] size.
      labels: Labels. 1D tensor of [batch_size] size.
    Raises:
      ValueError: If no data_dir
    """
    with tf.variable_scope('READ'):
        if not FLAGS.data_dir:
            raise ValueError('Please supply a data_dir')
        data_dir = os.path.join(FLAGS.data_dir, '')
        images, labels = align_input.inputs(eval_data=eval_data,
                                         data_dir=data_dir,
                                         batch_size=FLAGS.batch_size)

        if FLAGS.use_fp16:
            images = tf.cast(images, tf.float16)
            labels = tf.cast(labels, tf.float16)

        return images, labels

def dummy_layer(input, name='dummy_layer'):
    with tf.variable_scope(name):
        with tf.device('/cpu:0'):
            dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
            weights = tf.get_variable('weights', [1, 1], initializer=tf.constant_initializer(value=1), dtype=dtype)
            tf.add_to_collection('dummy', weights)
            return tf.matmul(input, weights)


def inference(images, eval=False):
    """
    Build the model in which firstly extract features from both input images first. Then concat them together

    :param images: Images reterned from distored_inputs() or inputs(), tensor_shape = [batch_size, width, height, 6]
    :return: Logits
    """
    tran_images = spatial_transformer_layer(images, eval=eval, name='spatial_transformer')
    tran_images = tf.sign(tran_images)
    area = tf.reduce_sum(tf.reshape(tran_images, [FLAGS.batch_size, -1]), axis=1)
    # dummy = tf.Variable(1.0)
    # tf.add_to_collection('dummy', dummy)
    # area = tf.multiply(dummy, area)
    return area
    # area = tf.reshape(area, [FLAGS.batch_size, 1])

    # out = dummy_layer(area)


def loss(logits, labels):
    logits = tf.cast(logits, tf.float32)
    labels = tf.cast(labels, tf.float32)
    l = tf.reduce_mean(tf.square(tf.divide(labels - logits, labels)))
    tf.summary.scalar('loss', l)
    return l

def train(total_loss, global_step):
    with tf.variable_scope('train_op'):
        num_batches_per_epoch = FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
        decay_steps = int(num_batches_per_epoch * FLAGS.NUM_EPOCHS_PER_DECAY)

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(FLAGS.INITIAL_LEARNING_RATE,
                                        global_step,
                                        decay_steps,
                                        FLAGS.LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)
        tf.summary.scalar('learning_rate', lr)

        train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(total_loss)

    return train_op

# def train(total_loss, global_step):
#     """
#     Train MSHAPES model.
#     Create an optimizer and apply to all trainable variables. Add moving
#     average for all trainable variables.
#
#     :param total_loss: Total loss from loss().
#     :param global_step: Integer Variable counting the number of training steps processed.
#     :return: op for training.
#     """
#     # Variables that affect learning rate.
#     with tf.variable_scope('train_op'):
#         num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
#         decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
#
#         # Decay the learning rate exponentially based on the number of steps.
#         lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
#                                         global_step,
#                                         decay_steps,
#                                         LEARNING_RATE_DECAY_FACTOR,
#                                         staircase=True)
#         tf.summary.scalar('learning_rate', lr)
#
#         # train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(total_loss)
#
#         # Generate moving averages of all losses and associated summaries.
#         loss_averages_op = _add_loss_summaries(total_loss)
#
#         # Compute gradients.
#         with tf.control_dependencies([loss_averages_op]):
#             opt = tf.train.AdamOptimizer(learning_rate=lr)
#             grads = opt.compute_gradients(total_loss)
#
#         # Apply gradients.
#         apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
#
#         # Add histograms for trainable variables.
#         for var in tf.trainable_variables():
#             tf.summary.histogram(var.op.name, var)
#
#         # Add histograms for gradients.
#         for grad, var in grads:
#             if grad is not None:
#                 tf.summary.histogram(var.op.name + '/gradients', grad)
#
#         # Track the moving averages of all trainable variables.
#         variable_averages = tf.train.ExponentialMovingAverage(
#             MOVING_AVERAGE_DECAY, global_step)
#         variables_averages_op = variable_averages.apply(tf.trainable_variables())
#
#         with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
#             train_op = tf.no_op(name='train')
#
#     return train_op


def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.
    Args:
      x: Tensor
    Returns:
      nothing


    author: The TensorFlow Authors
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',
                      tf.nn.zero_fraction(x))



def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses)

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op



def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var



def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    Returns:
      Variable Tensor
    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def my_fake_variable(name, shape):
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=tf.constant_initializer(value=1), dtype=dtype)
    return var