
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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
            tf.add_to_collection('test', grid)
            tf.add_to_collection('test', A_theta)

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

if __name__ == '__main__':
    org_img = cv2.imread('./0.png', cv2.IMREAD_GRAYSCALE)
    print org_img.shape
    plt.figure('origin')
    plt.imshow(org_img)
    # cv2.imshow('origin', org_img)

    img = np.reshape(org_img, [1,200,200,1])
    img = tf.constant(img)
    theta = tf.constant([[0.5,0.5,0.5]])
    tran_img = rotate_and_translation_transformer(img, theta, (200,200))

    sess = tf.Session()
    tran_img = sess.run(tran_img)
    grid, A_theta = sess.run(tf.get_collection('test'))
    print np.array(grid)
    print np.array(A_theta)
    tran_img = np.reshape(tran_img, [200, 200])

    plt.figure('trans')
    plt.imshow(tran_img)
    plt.show()