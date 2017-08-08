import tensorflow as tf

def inference(input_tensor, eval=False, name='localisation_net', output_size=1):

    CONV1_DEPTH = 64
    CONV2_DEPTH = 96
    FC1_SIZE = 128

    batch_size = input_tensor.get_shape().as_list()[0]

    with tf.variable_scope(name):
        with tf.variable_scope('conv1'):
            conv1 = tf.layers.conv2d(input_tensor,
                                     filters=CONV1_DEPTH,
                                     kernel_size=(5, 5),
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
        with tf.variable_scope('fc1'):
            flatten = tf.reshape(pool2, shape=(batch_size, -1))
            fc1 = tf.layers.dense(flatten,
                                  units=FC1_SIZE,
                                  activation=tf.nn.relu,
                                  use_bias=True,
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                  bias_initializer=tf.constant_initializer(0.1),
                                  kernel_regularizer=tf.nn.l2_loss)
        with tf.variable_scope('output'):
            fc2 = tf.layers.dense(fc1,
                                  units=output_size,
                                  activation=tf.nn.relu,
                                  use_bias=True,
                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                  bias_initializer=tf.constant_initializer(0.1),
                                  kernel_regularizer=tf.nn.l2_loss)
    return fc2