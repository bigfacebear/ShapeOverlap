import os
import sys

import tensorflow as tf

import models
from overlap_train import train
import FLAGS

train_dir = os.path.join(FLAGS.train_dir, 'siamese')

def main(argv=None):

    print 'Begin training...'
    sys.stdout.flush()
    train(train_dir, models.siamese_inference)
    print 'Finished'


if __name__ == '__main__':
    tf.app.run()