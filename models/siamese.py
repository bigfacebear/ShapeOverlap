
import tensorflow as tf

# model parameters
CONV1_DEPTH = 96/2
CONV2_DEPTH = 192/2
CONV3_DEPTH = 256/2

FC1_SIZE = 512/2
FC2_SIZE = 512/2
OUTPUT_SIZE = 1

def inference(locks, keys, eval=False):
    # Merge locks batch and keys batch together
    # e.g., [128, 150, 150, 1] merge [128, 150, 150, 1] -> [256, 150, 150, 1]
    merge = tf.concat((locks, keys), axis=0)
    batch_size = merge.get_shape().as_list()[0]

    with tf.variable_scope('conv1'):
        conv1 = tf.layers.conv2d(merge,
                                 filters=CONV1_DEPTH,
                                 kernel_size=(3, 3),
                                 strides=(1, 1),
                                 padding='valid',
                                 activation=tf.nn.relu,
                                 use_bias=True,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 bias_initializer=tf.constant_initializer(0.1))
    with tf.variable_scope('pool1'):
        pool1 = tf.layers.average_pooling2d(conv1,
                                            pool_size=(2, 2),
                                            strides=(2, 2),
                                            padding='valid')
    with tf.variable_scope('conv2'):
        conv2 = tf.layers.conv2d(pool1,
                                 filters=CONV2_DEPTH,
                                 kernel_size=(5, 5),
                                 strides=(1, 1),
                                 padding='valid',
                                 activation=tf.nn.relu,
                                 use_bias=True,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 bias_initializer=tf.constant_initializer(0.1))
    with tf.variable_scope('pool2'):
        pool2 = tf.layers.average_pooling2d(conv2,
                                            pool_size=(2, 2),
                                            strides=(2, 2),
                                            padding='valid')
    with tf.variable_scope('conv3'):
        conv3 = tf.layers.conv2d(pool2,
                                 filters=CONV3_DEPTH,
                                 kernel_size=(3, 3),
                                 strides=(1, 1),
                                 padding='valid',
                                 activation=tf.nn.relu,
                                 use_bias=True,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 bias_initializer=tf.constant_initializer(0.1))
    with tf.variable_scope('fc1'):
        flatten = tf.reshape(conv3, shape=(batch_size, -1))
        fc1 = tf.layers.dense(flatten,
                              units=FC1_SIZE,
                              activation=tf.nn.relu,
                              use_bias=True,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                              bias_initializer=tf.constant_initializer(0.1),
                              kernel_regularizer=tf.nn.l2_loss)
    with tf.variable_scope('fc2'):
        fc2 = tf.layers.dense(fc1,
                              units=FC2_SIZE,
                              activation=tf.nn.relu,
                              use_bias=True,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                              bias_initializer=tf.constant_initializer(0.1),
                              kernel_regularizer=tf.nn.l2_loss)
    with tf.variable_scope('output'):
        fc3 = tf.layers.dense(fc2,
                              units=OUTPUT_SIZE,
                              activation=tf.nn.relu,
                              use_bias=True,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                              bias_initializer=tf.constant_initializer(0.1),
                              kernel_regularizer=tf.nn.l2_loss)

    return fc3