#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main file for MSHAPES train
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime

import tensorflow as tf
import numpy as np

import align
import align_FLAGS as FLAGS

def train():
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        with tf.device('/cpu:0'):
            images, labels = align.inputs(eval_data=False)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = align.inference(images)

        # Calculate loss.
        loss = align.loss(logits, labels)
        tf.summary.scalar('loss', loss)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = align.train(loss, global_step)

        saver = tf.train.Saver(var_list=(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))
        summary_op_merged = tf.summary.merge_all()

        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter(align.FLAGS.train_dir, sess.graph)
            tf.set_random_seed(42)
            tf.global_variables_initializer().run()

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)

            start_time = time.time()

            dummy_refresh_op = tf.variables_initializer(tf.get_collection('dummy'))
            for i in xrange(FLAGS.max_steps):

                dummy_refresh_op.run()

                # area, l = sess.run([logits, loss])
                #
                # print('area', area.tolist())
                # print('loss', l.tolist())
                #
                # saver.save(sess, os.path.join(FLAGS.train_dir, 'area_train.ckpt'))  # , global_step=i)
                # summary_str = sess.run(summary_op_merged)
                # train_writer.add_summary(summary_str, i)


                _, my_loss = sess.run([train_op, loss])
                ml = np.array(my_loss)

                if (i + 1) % FLAGS.log_frequency == 0 and i != 0:  # Every 1000 steps, save the results and send an email
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

                    saver.save(sess, os.path.join(FLAGS.train_dir, 'area_train.ckpt'))  # , global_step=i)
                    summary_str = sess.run(summary_op_merged)
                    train_writer.add_summary(summary_str, i)

            coord.request_stop()
            coord.join(threads)

        print('Finished.')


def main(argv=None):
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)

    # Train the network!!
    print('Begin training...')
    train()



if __name__ == '__main__':
    tf.app.run()