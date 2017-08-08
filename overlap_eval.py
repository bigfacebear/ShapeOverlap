from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import overlap
import models

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', './ShapeOverlap_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './ShapeOverlap_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 20,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 20000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Batch size.""")

def eval_once(saver, summary_writer, diff, summary_op):
    """Run Eval once.

    Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
    """
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        print "checkpoint dir =", ckpt.model_checkpoint_path
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print "global step =", global_step
        else:
            print 'No checkpoint file found'
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        threads = []
        try:
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                start=True))

            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            results = []
            while step < num_iter and not coord.should_stop():
                predictions = sess.run([diff])
                results += predictions[0].tolist()
                step += 1

            # Compute precision @ 1.
            print len(results)
            print results
            results = np.array(results)
            average = np.average(results)
            variation = np.var(results)
            print '%s: average difference = %.3f, variation of difference = %.3f' % (datetime.now(), average, variation)

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='average difference', simple_value=average)
            summary.value.add(tag='variation', simple_value=variation)
            summary_writer.add_summary(summary, global_step=1)
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads)#, stop_grace_period_secs=10)


def evaluate():
    with tf.Graph().as_default() as g:
        eval_data = FLAGS.eval_data == 'test'
        locks, keys, labels = overlap.inputs(eval_data=eval_data)

        logits = models.siamese_inference(locks, keys, eval=True)

        diff = tf.abs(logits - tf.cast(labels, tf.float32))

        saver = tf.train.Saver()

        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

        while True:
            eval_once(saver, summary_writer, diff, summary_op)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()


if __name__ == '__main__':
    tf.app.run()