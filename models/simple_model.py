
import tensorflow as tf

from model_utils import _variable_with_weight_decay, _variable_on_cpu

# model parameters
NUM_CLASSES = 1
CONVOLUTIONAL_LAYER_DEPTH = 16
FC1_NUM = 786
FC2_NUM = 384
KEEP_PROB = 0.5

def image_process(name, images, eval=False):
    CONV1_DEPTH = CONVOLUTIONAL_LAYER_DEPTH
    CONV2_DEPTH = CONVOLUTIONAL_LAYER_DEPTH
    CONV3_DEPTH = CONVOLUTIONAL_LAYER_DEPTH

    channel_num = images.get_shape().as_list()[3]

    with tf.variable_scope(name):

        # conv1
        with tf.variable_scope('conv1') as scope:
            kernel = _variable_with_weight_decay('weights',
                                                 shape=[5, 5, channel_num, CONV1_DEPTH],
                                                 stddev=5e-3,
                                                 wd=0.0)
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu('biases', [CONV1_DEPTH], tf.constant_initializer(1e-2))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(pre_activation, name=scope.name)

            # pool1
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                                   padding='SAME', name='pool')
            # norm1
            norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                              name='norm1')

        # conv2
        with tf.variable_scope('conv2') as scope:
            kernel = _variable_with_weight_decay('weights',
                                                 shape=[5, 5, CONV1_DEPTH, CONV2_DEPTH],
                                                 stddev=5e-2,
                                                 wd=0.0)
            conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu('biases', [CONV2_DEPTH], tf.constant_initializer(0.1))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(pre_activation, name=scope.name)

            # norm2
            norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                              name='norm2')
            # pool2
            pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                                   strides=[1, 2, 2, 1], padding='SAME', name='pool1')

        # conv3
        with tf.variable_scope('conv3') as scope:
            kernel = _variable_with_weight_decay('weights',
                                                 shape=[5, 5, CONV2_DEPTH, CONV3_DEPTH],
                                                 stddev=5e-2,
                                                 wd=0.0)
            conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu('biases', [CONV3_DEPTH], tf.constant_initializer(0.1))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv3 = tf.nn.relu(pre_activation, name=scope.name)

            # norm2
            norm3 = tf.nn.lrn(conv3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                              name='norm3')
            # pool2
            pool3 = tf.nn.max_pool(norm3, ksize=[1, 3, 3, 1],
                                   strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    return pool3

def fully_connected_layer(features, eval=False):
    batch_size = features.get_shape().as_list()[0]

    # FC1
    with tf.variable_scope('FC1') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(features, [batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, FC1_NUM],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [FC1_NUM], tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        keep_prob = KEEP_PROB if eval else 1.0
        fc1_dropout = tf.nn.dropout(fc1, keep_prob=keep_prob)

    # FC2
    with tf.variable_scope('FC2') as scope:
        weights = _variable_with_weight_decay('weights', shape=[FC1_NUM, FC2_NUM],  # 192
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [FC2_NUM], tf.constant_initializer(0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1_dropout, weights) + biases, name=scope.name)

    # linear layer(WX + b),
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
    with tf.variable_scope('regression') as scope:
        weights = _variable_with_weight_decay('weights', [FC2_NUM, NUM_CLASSES],
                                              stddev=1 / float(FC2_NUM), wd=0.0)
        biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                  tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(fc2, weights), biases, name=scope.name)

    return softmax_linear

def inference(locks, keys, eval=False):
    batch_size = locks.get_shape().as_list()[0]
    L_features = image_process('locks', locks, eval=eval)
    K_features = image_process('keys', keys, eval=eval)
    features = tf.concat([L_features, K_features], axis=3)
    ret = fully_connected_layer(features, eval)
    ret = tf.reduce_sum(ret, axis=1)
    return tf.reshape(ret, [batch_size])