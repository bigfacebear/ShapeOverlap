import time
from datetime import datetime
import os
import sys

import tensorflow as tf

import overlap
import FLAGS

def train(train_dir, inference):
    if tf.gfile.Exists(train_dir):
        tf.gfile.DeleteRecursively(train_dir)
    tf.gfile.MakeDirs(train_dir)

    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        with tf.device('/cpu:0'):
            locks, keys, labels = overlap.inputs(eval_data=False)

        logits = inference(locks, keys)

        loss = overlap.loss(logits, labels)

        train_op = overlap.train(loss, global_step)

        saver = tf.train.Saver(var_list=(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))
        summary_op_merged = tf.summary.merge_all()

        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter(train_dir, sess.graph)
            tf.set_random_seed(42)
            tf.global_variables_initializer().run()

            # the initialize op to re-initialize the dummy node in every training step
            dummy_init_op = tf.variables_initializer(tf.get_collection('dummy'))

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)

            start_time = time.time()
            for i in xrange(FLAGS.max_steps):
                dummy_init_op.run() # re-initialize the dummy node (to 1.0)
                _, my_loss = sess.run([train_op, loss])

                if (i + 1) % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - start_time
                    start_time = current_time

                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print format_str % (datetime.now(), (i + 1), my_loss,
                                        examples_per_sec, sec_per_batch)
                    sys.stdout.flush()

                    saver.save(sess, os.path.join(train_dir, 'ShapeOverlap_train.ckpt'))
                    summary_str = sess.run(summary_op_merged)
                    train_writer.add_summary(summary_str, i)

            coord.request_stop()
            coord.join(threads)