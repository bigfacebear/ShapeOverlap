import os
import sys

import tensorflow as tf

import models
from overlap_train import train
import FLAGS

train_dir = os.path.join(FLAGS.train_dir, 'transformer_matching_aff')

def inference(locks, keys, eval=False):
    batch_size, height, width, _ = locks.get_shape().as_list()
    theta = models.matching_inference(locks, keys, eval, output_size=6)
    keys = models.affine_transformer(keys, theta, (height, width))
    keys = tf.sign(keys)  # convert to 0s and 1s
    locks = tf.sign(locks)
    overlap = tf.multiply(keys, locks)  # calculate overlap
    area = tf.reduce_sum(tf.reshape(overlap, (batch_size, -1)), axis=1)
    return area

def main(argv=None):

    print 'Begin training...'
    sys.stdout.flush()
    train(train_dir, inference)
    print 'Finished'


if __name__ == '__main__':
    tf.app.run()