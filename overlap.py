import os
import re

import tensorflow as tf

import FLAGS
import overlap_input
import models
from utils import maybe_download_and_extract


def inputs(eval_data):
    """Construct input for ShapeOverlap evaluation using the Reader ops.
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
        locks, keys, labels = overlap_input.inputs(eval_data=eval_data,
                                         data_dir=FLAGS.data_dir,
                                         batch_size=FLAGS.batch_size)

        if FLAGS.use_fp16:
            locks = tf.cast(locks, tf.float16)
            keys = tf.cast(keys, tf.float16)
            labels = tf.cast(labels, tf.float16)

        return locks, keys, labels

def loss(logits, labels):
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