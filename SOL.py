from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

import tensorflow as tf

import FLAGS
import SOL_input
from spatial_transformer import affine_transformer, rotate_and_translation_transformer
from utils import maybe_download_and_extract

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
    maybe_download_and_extract(FLAGS.data_dir, FLAGS.DATA_URL)

    with tf.variable_scope('READ'):
        if not FLAGS.data_dir:
            raise ValueError('Please supply a data_dir')
        locks, keys, labels = SOL_input.inputs(eval_data=eval_data,
                                         data_dir=FLAGS.data_dir,
                                         batch_size=FLAGS.batch_size)

        if FLAGS.use_fp16:
            locks = tf.cast(locks, tf.float16)
            keys = tf.cast(keys, tf.float16)
            labels = tf.cast(labels, tf.float16)

        return locks, keys, labels


def image_process(name, images, eval=False):
    CONV1_DEPTH = FLAGS.CONVOLUTIONAL_LAYER_DEPTH
    CONV2_DEPTH = FLAGS.CONVOLUTIONAL_LAYER_DEPTH
    CONV3_DEPTH = FLAGS.CONVOLUTIONAL_LAYER_DEPTH

    channel_num = images.get_shape().as_list()[3]

    with tf.variable_scope(name):
        trans = spatial_transformer_layer(images, eval=eval)

        # conv1
        with tf.variable_scope('conv1') as scope:
            kernel = _variable_with_weight_decay('weights',
                                                 shape=[5, 5, channel_num, CONV1_DEPTH],
                                                 stddev=5e-3,
                                                 wd=0.0)
            conv = tf.nn.conv2d(trans, kernel, [1, 1, 1, 1], padding='SAME')
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

        # conv3
        with tf.variable_scope('conv3') as scope:
            kernel = _variable_with_weight_decay('weights',
                                                 shape=[5, 5, CONV2_DEPTH, CONV3_DEPTH],
                                                 stddev=5e-2,
                                                 wd=0.0)
            conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu('biases', [CONV3_DEPTH], tf.constant_initializer(0.1))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv3 = tf.nn.relu(pre_activation, name=scope.name)

            # norm2
            norm3 = tf.nn.lrn(conv3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                              name='norm3')
            # pool2
            pool3 = tf.nn.max_pool(norm3, ksize=[1, 3, 3, 1],
                                   strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    return pool3

def full_connection_layer(features, eval=False):
    FC1_NUM = 1536
    FC2_NUM = 384
    # FC1
    with tf.variable_scope('FC1') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(features, [FLAGS.batch_size, -1])
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
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
    with tf.variable_scope('regression') as scope:
        weights = _variable_with_weight_decay('weights', [FC2_NUM, NUM_CLASSES],
                                              stddev=1 / float(FC2_NUM), wd=0.0)
        biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                  tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(fc2, weights), biases, name=scope.name)

    return softmax_linear

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
            biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                      tf.constant_initializer(0.0))
            theta = tf.add(tf.matmul(fc2, weights), biases, name=scope.name)

    return theta


def spatial_transformer_layer(input_tensor, eval=False, name='spatial_transformer'):
    with tf.variable_scope(name):
        shape = input_tensor.get_shape().as_list()
        height, width = shape[1], shape[2]
        theta = localisation_net(input_tensor, eval=eval)
        return rotate_and_translation_transformer(input_tensor, theta, (height, width))


def inference(locks, keys, eval=False):
    L_features = image_process('locks', locks, eval=eval)
    K_features = image_process('keys', keys, eval=eval)
    features = tf.concat([L_features, K_features], axis=3)
    ret = full_connection_layer(features, eval)
    return tf.reshape(ret, [FLAGS.batch_size])


def st_inference(locks, keys, eval=False):
    st_locks = spatial_transformer_layer(locks, eval=eval, name='st_locks')
    st_keys = spatial_transformer_layer(keys, eval=eval, name='st_keys')
    st_locks_bool = tf.cast(st_locks, dtype=tf.bool)
    st_keys_bool = tf.cast(st_keys, dtype=tf.bool)
    overlap = tf.logical_and(st_locks_bool, st_keys_bool)
    # overlap = tf.concat([st_locks, st_keys], axis=3)
    overlap = tf.cast(overlap, dtype=tf.float32)
    nonzeros = tf.reduce_sum(overlap, axis=1)
    # nonzeros = tf.count_nonzero(tf.reshape(overlap, [FLAGS.batch_size, -1]), axis=1)
    # ret = full_connection_layer(overlap, eval=eval)
    return nonzeros

    # print('overlap', overlap.get_shape())
    # overlap = tf.reshape(tf.cast(overlap, dtype=tf.int32), [FLAGS.batch_size, -1])
    # print('overlap', overlap.get_shape())
    # nonzeros = tf.reduce_sum(overlap, axis=1)
    # print('nonzeros shape', nonzeros.get_shape())
    # return tf.reshape(nonzeros, [FLAGS.batch_size])
    # nonzeros = tf.count_nonzero(tf.reshape(overlap, [FLAGS.batch_size, -1]), axis=1)
    # print('nonzeros shape', nonzeros.get_shape())
    # return nonzeros

def loss(logits, labels):
    labels = tf.cast(labels, tf.float32)
    logits = tf.cast(logits, tf.float32)
    l = tf.reduce_mean(tf.square(logits - labels))
    tf.add_to_collection('losses', l)
    return l

def train(total_loss, global_step):
    with tf.variable_scope('train_op'):
        num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
        decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                        global_step,
                                        decay_steps,
                                        LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)
        tf.summary.scalar('learning_rate', lr)

        # train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(total_loss)
        # return train_op

        # Generate moving averages of all losses and associated summaries.
        loss_averages_op = _add_loss_summaries(total_loss)

        # Compute gradients.
        with tf.control_dependencies([loss_averages_op]):
            opt = tf.train.AdamOptimizer(learning_rate=lr)
            grads = opt.compute_gradients(total_loss)

        # Apply gradients.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # # Add histograms for trainable variables.
        # for var in tf.trainable_variables():
        #     tf.summary.histogram(var.op.name, var)

        # Add histograms for gradients.
        # for grad, var in grads:
        #     if grad is not None:
        #         tf.summary.histogram(var.op.name + '/gradients', grad)

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train')

    return train_op


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
    # loss_averages_op = loss_averages.apply(losses + [total_loss])
    loss_averages_op = loss_averages.apply(losses)

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    # for l in losses + [total_loss]:
    #     # Name each loss as '(raw)' and name the moving average version of the loss
    #     # as the original loss name.
    #     tf.summary.scalar(l.op.name + ' (raw)', l)
    #     tf.summary.scalar(l.op.name, loss_averages.average(l))

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
