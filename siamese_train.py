from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from datetime import datetime
import os
import sys

import tensorflow as tf
import numpy as np

import SOL
import FLAGS
import models

train_dir = '/cstor/xsede/users/xs-qczhao/train/ShapeOverlap_siamese_train'
# train_dir = './ShapeOverlap_train'

def train():
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        with tf.device('/cpu:0'):
            locks, keys, labels = SOL.inputs(eval_data=False)

        # Build a graph that computes the logits predictions from the inference model
        logits = models.siamese_inference(locks, keys)

        # Calculate loss.
        loss = SOL.loss(logits, labels)
        tf.summary.scalar('loss', loss)

        # Build a graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = SOL.train(loss, global_step)

        saver = tf.train.Saver(var_list=(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))
        summary_op_merged = tf.summary.merge_all()

        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter(train_dir, sess.graph)
            tf.set_random_seed(42)
            tf.global_variables_initializer().run()

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)

            start_time = time.time()
            for i in xrange(FLAGS.max_steps):
                _, my_loss = sess.run([train_op, loss])
                ml = np.array(my_loss)

                if (
                    i + 1) % FLAGS.log_frequency == 0 and i != 0:  # Every 1000 steps, save the results and send an email
                    current_time = time.time()
                    duration = current_time - start_time
                    start_time = current_time

                    loss_value = ml
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), (i + 1), loss_value,
                                        examples_per_sec, sec_per_batch))
                    sys.stdout.flush()

                    saver.save(sess, os.path.join(train_dir, 'ShapeOverlap_train.ckpt'))  # , global_step=i)
                    summary_str = sess.run(summary_op_merged)
                    train_writer.add_summary(summary_str, i)

            coord.request_stop()
            coord.join(threads)

            print('Finished.')

def main(argv=None):
    if tf.gfile.Exists(train_dir):
        tf.gfile.DeleteRecursively(train_dir)
    tf.gfile.MakeDirs(train_dir)

    # Train the network!!
    print('Begin training...')
    train()


if __name__ == '__main__':
    tf.app.run()