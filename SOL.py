from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

import tensorflow as tf

import FLAGS
import SOL_input
import models
from utils import maybe_download_and_extract

# Global constants describing the MSHAPES data set.
IMAGE_SIZE = FLAGS.IMAGE_SIZE
NUM_CLASSES = FLAGS.NUM_CLASSES
# NUM_CLASSES = 5
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


def inference(locks, keys, eval=False):
    batch_size = locks.get_shape().as_list()[0]
    ret = models.siamese_inference(locks, keys, eval)
    return tf.reshape(ret, [batch_size])


# def st_inference(locks, keys, eval=False):
#     st_locks = spatial_transformer_layer(locks, eval=eval, name='st_locks')
#     st_keys = spatial_transformer_layer(keys, eval=eval, name='st_keys')
#     st_locks_bool = tf.cast(st_locks, dtype=tf.bool)
#     st_keys_bool = tf.cast(st_keys, dtype=tf.bool)
#     overlap = tf.logical_and(st_locks_bool, st_keys_bool)
#     overlap = tf.cast(overlap, dtype=tf.float32)
#     area = tf.reduce_sum(tf.reshape(overlap, [FLAGS.batch_size, -1]), axis=1)
#     area = tf.reshape(area, [FLAGS.batch_size, 1])
#     return tf.reshape(dummy_layer(area), [FLAGS.batch_size])
#
#     # print('overlap', overlap.get_shape())
#     # overlap = tf.reshape(tf.cast(overlap, dtype=tf.int32), [FLAGS.batch_size, -1])
#     # print('overlap', overlap.get_shape())
#     # nonzeros = tf.reduce_sum(overlap, axis=1)
#     # print('nonzeros shape', nonzeros.get_shape())
#     # return tf.reshape(nonzeros, [FLAGS.batch_size])
#     # nonzeros = tf.count_nonzero(tf.reshape(overlap, [FLAGS.batch_size, -1]), axis=1)
#     # print('nonzeros shape', nonzeros.get_shape())
#     # return nonzeros
#
# def inference_mathing_layer(locks, keys, eval=False):
#     locks = spatial_transformer_layer(locks, eval=eval, name='locks')
#     locks_pool = tf.nn.avg_pool(locks, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='locks_pool')
#     keys = spatial_transformer_layer(keys, eval=eval, name='keys')
#     keys_pool = tf.nn.avg_pool(keys, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='locks_pool')
#     matched = matching_layer(locks_pool, keys_pool, name='matching')
#     return fully_connected_layer(matched, eval=eval)


def loss(logits, labels):
    labels = tf.cast(labels, tf.float32)
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
