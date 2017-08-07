import tensorflow as tf

import FLAGS


def rotate_and_translation_transformer(U, theta, out_size, name='rotate_and_translation_transformer'):
    def _interpolate(im, x, y, out_size):
        with tf.variable_scope('_interpolate'):
            # constants
            im_shape = im.get_shape().as_list()
            num_batch = im_shape[0]
            height = im_shape[1]
            width = im_shape[2]
            channels = im_shape[3]

            x = tf.cast(x, dtype=tf.float32)
            y = tf.cast(y, dtype=tf.float32)
            height_f = tf.cast(height, dtype=tf.float32)
            width_f = tf.cast(width, dtype=tf.float32)
            out_height = out_size[0]
            out_width = out_size[1]
            zero = tf.zeros([], dtype=tf.int32)
            max_y = tf.cast(tf.shape(im)[1] - 1, dtype=tf.int32)
            max_x = tf.cast(tf.shape(im)[2] - 1, dtype=tf.int32)

            # scale indices from [-1, 1] to [0, width/height]
            # nearest
            _x = tf.cast((x + 1.0) * (width_f) / 2.0 + 0.5, dtype=tf.int32)
            _y = tf.cast((y + 1.0) * (height_f) / 2.0 + 0.5, dtype=tf.int32)
            _x = tf.clip_by_value(_x, zero, max_x)
            _y = tf.clip_by_value(_y, zero, max_y)

            batch_range = tf.range(num_batch) * width * height
            base_batch = tf.reshape(tf.stack([batch_range for _ in range(out_height * out_width)], axis=1), [-1])
            base_y = base_batch + _y * width
            idx = base_y + _x
            im_flat = tf.reshape(im, tf.stack([-1, channels]))
            output = tf.gather(im_flat, idx)
            return output

    def _meshgrid(height, width):
        with tf.variable_scope('_meshgrid'):
            # This should be equivalent to:
            #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
            #                         np.linspace(-1, 1, height))
            #  ones = np.ones(np.prod(x_t.shape))
            #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])

            # shape of x_t, y_t is [height, width]
            x_t = tf.matmul(tf.ones(shape=[height, 1]),
                            tf.expand_dims(tf.linspace(-1.0, 1.0, width), 0))
            y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                            tf.ones(shape=[1, width]))

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))

            ones = tf.ones_like(x_t_flat)
            grid = tf.concat(axis=0, values=[x_t_flat, y_t_flat, ones])
            return grid

    def _transform(theta, input_dim, out_size):
        with tf.variable_scope('_transform'):
            input_shape = input_dim.get_shape().as_list()
            num_batch = input_shape[0]
            num_channels = input_shape[3]

            # theta is [Tx, Ty, a], a is the rotate angle
            # A_theta = [
            # cos_a    -sin_a    Tx
            # sin_a    cos_a     Ty
            # ]
            theta = tf.cast(theta, dtype=tf.float32)
            angle = theta[:, 2]
            sin_a = tf.sin(angle)
            cos_a = tf.cos(angle)
            A_theta = tf.stack([cos_a, -sin_a, theta[:, 0], sin_a, cos_a, theta[:, 1]], axis=1)
            A_theta = tf.reshape(A_theta, shape=[num_batch, 2, 3])

            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            out_height = out_size[0]
            out_width = out_size[1]
            grid = _meshgrid(out_height, out_width)
            grid = tf.reshape(grid, [-1])
            grid = tf.tile(grid, tf.stack([num_batch]))
            grid = tf.reshape(grid, [num_batch, 3, -1])

            # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
            T_g = tf.matmul(A_theta, grid)
            x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
            y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
            x_s_flat = tf.reshape(x_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])

            input_transformed = _interpolate(
                input_dim, x_s_flat, y_s_flat,
                out_size)

            output = tf.reshape(
                input_transformed, [num_batch, out_height, out_width, num_channels])
            return output

    with tf.variable_scope(name):
        output = _transform(theta, U, out_size)
        return output



def localisation_net(input_tensor, eval=False, name='localisation_net'):
    CONV1_DEPTH = 16
    CONV2_DEPTH = 16

    FC1_NUM = 786
    FC2_NUM = 384

    theta_param_num = 3

    channel_num = input_tensor.get_shape().as_list()[3]
    with tf.variable_scope(name):
        # conv1
        with tf.variable_scope('conv1') as scope:
            kernel = _variable_with_weight_decay('weights',
                                                 shape=[5, 5, channel_num, CONV1_DEPTH],
                                                 stddev=5e-3,
                                                 wd=0.0)
            conv = tf.nn.conv2d(input_tensor, kernel, [1, 1, 1, 1], padding='SAME')
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
        # FC1
        with tf.variable_scope('FC1') as scope:
            # Move everything into depth so we can perform a single matrix multiply.
            reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
            dim = reshape.get_shape()[1].value
            weights = _variable_with_weight_decay('weights', shape=[dim, FC1_NUM],
                                                  stddev=0.04, wd=0.004)
            biases = _variable_on_cpu('biases', [FC1_NUM], tf.constant_initializer(0.1))
            fc1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
            keep_prob = FLAGS.KEEP_PROB if eval else 1.0
            fc1_dropout = tf.nn.dropout(fc1, keep_prob=keep_prob)

        # FC2
        with tf.variable_scope('FC2') as scope:
            weights = _variable_with_weight_decay('weights', shape=[FC1_NUM, FC2_NUM],  # 192
                                                  stddev=0.04, wd=0.004)
            biases = _variable_on_cpu('biases', [FC2_NUM], tf.constant_initializer(0.1))
            fc2 = tf.nn.relu(tf.matmul(fc1_dropout, weights) + biases, name=scope.name)

        # linear layer(WX + b),
        # and performs the softmax internally for efficiency.
        with tf.variable_scope('theta') as scope:
            weights = _variable_with_weight_decay('weights', [FC2_NUM, theta_param_num],
                                                  stddev=1 / float(FC2_NUM), wd=0.0)
            biases = _variable_on_cpu('biases', [theta_param_num],
                                      tf.constant_initializer(0.0))
            theta = tf.add(tf.matmul(fc2, weights), biases, name=scope.name)

    return theta


def spatial_transformer_layer(input_tensor, eval=False, name='spatial_transformer'):
    with tf.variable_scope(name):
        shape = input_tensor.get_shape().as_list()
        height, width = shape[1], shape[2]
        theta = localisation_net(input_tensor, eval=eval)
        return rotate_and_translation_transformer(input_tensor, theta, (height * 2 // 3, width * 2 // 3))


def image_process(name, images, eval=False):
    CONV1_DEPTH = FLAGS.CONVOLUTIONAL_LAYER_DEPTH
    CONV2_DEPTH = FLAGS.CONVOLUTIONAL_LAYER_DEPTH
    CONV3_DEPTH = FLAGS.CONVOLUTIONAL_LAYER_DEPTH

    channel_num = images.get_shape().as_list()[3]

    with tf.variable_scope(name):
        trans = spatial_transformer_layer(images, eval=eval)

        # conv1
        with tf.variable_scope('conv1') as scope:
            kernel = _variable_with_weight_decay('weights',
                                                 shape=[5, 5, channel_num, CONV1_DEPTH],
                                                 stddev=5e-3,
                                                 wd=0.0)
            conv = tf.nn.conv2d(trans, kernel, [1, 1, 1, 1], padding='SAME')
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
    FC1_NUM = 786
    FC2_NUM = 384
    # FC1
    with tf.variable_scope('FC1') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(features, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, FC1_NUM],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [FC1_NUM], tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        keep_prob = FLAGS.KEEP_PROB if eval else 1.0
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

def matching_layer(L, K, name='matching_layer'):
        with tf.variable_scope(name) as scope:
            bs, h, w, d = L.get_shape().as_list()
            L = tf.reshape(L, [bs, h * w, d])
            K = tf.reshape(K, [bs, h * w, d])
            K = tf.transpose(K, [0, 2, 1])
            return tf.reshape(tf.matmul(L, K), [bs, h, w, h * w])

def dummy_layer(input, name='dummy_layer'):
    with tf.variable_scope(name):
        with tf.device('/cpu:0'):
            dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
            weights = tf.get_variable('weights', [1, 1], initializer=tf.constant_initializer(value=1), dtype=dtype)
        return tf.matmul(input, weights)






