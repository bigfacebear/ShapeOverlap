
import tensorflow as tf

# model parameters
CONV1_DEPTH = 96/2
CONV2_DEPTH = 192/2
CONV3_DEPTH = 256/2

FC1_SIZE = 256/2
OUTPUT_SIZE = 1

def inference(locks, keys, eval=False):
    # concat two inputs
    concat = tf.concat([locks, keys], axis=3)
    batch_size = concat.get_shape().as_list()[0]

    with tf.variable_scope('conv1'):
        conv1 = tf.layers.conv2d(concat,
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
                                 kernel_size=(3, 3),
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
    with tf.variable_scope('output'):
        fc2 = tf.layers.dense(fc1,
                              units=OUTPUT_SIZE,
                              activation=tf.nn.relu,
                              use_bias=True,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                              bias_initializer=tf.constant_initializer(0.1),
                              kernel_regularizer=tf.nn.l2_loss)

    return fc2