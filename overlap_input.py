import os

import pickle
import tensorflow as tf
import FLAGS

IMAGE_SIZE = FLAGS.IMAGE_SIZE
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = FLAGS.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


def read_pair(filename_queue):
    lock_image = decode_input(filename_queue[0])
    key_image = decode_input(filename_queue[1])
    overlap_area = tf.cast(filename_queue[2], tf.float32)
    return lock_image, key_image, overlap_area


def decode_input(file_path):
    serialized_record = tf.read_file(file_path)
    image = tf.image.decode_png(serialized_record, dtype=tf.uint8, channels=1)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 1])
    return image



def inputs(eval_data, data_dir, batch_size):
    """
    Construct input for CIFAR evaluation using the Reader ops.
    :param eval_data: bool, indicating if one should use the train or eval data set.
    :param data_dir: Path to the ShapeOverlap data directory.
    :param batch_size: Number of images per batch.
    :return:
        locks: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1] size.
        keys: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1] size.
        labels: Labels. 1D tensor of [batch_size] size.
    """

    overlap_areas_filename = os.path.join(data_dir, 'OVERLAP_AREAS')
    if not tf.gfile.Exists(overlap_areas_filename):
        raise ValueError('Failed to find file: ' + overlap_areas_filename)
    with open(overlap_areas_filename) as fp:
        overlap_areas_list = pickle.load(fp)

    if not eval_data:
        lock_filenames = [os.path.join(data_dir, '%d_L.png' % i)
                          for i in xrange(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)]
        key_filenames = [os.path.join(data_dir, '%d_K.png' % i)
                         for i in xrange(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)]
        overlap_areas = overlap_areas_list[0:NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        lock_filenames = [os.path.join(data_dir, '%d_L.png' % i)
                          for i in xrange(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN,
                                          NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN + NUM_EXAMPLES_PER_EPOCH_FOR_EVAL)]
        key_filenames = [os.path.join(data_dir, '%d_K.png' % i)
                         for i in xrange(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN,
                                         NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN + NUM_EXAMPLES_PER_EPOCH_FOR_EVAL)]
        overlap_areas = overlap_areas_list[NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN:NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN+NUM_EXAMPLES_PER_EPOCH_FOR_EVAL]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
    for q in [lock_filenames, key_filenames, [os.path.join(data_dir, 'OVERLAP_AREAS')]]:
        for f in q:
            if not tf.gfile.Exists(f):
                raise ValueError('Failed to find file: ' + f)

    filename_queue = tf.train.slice_input_producer([lock_filenames, key_filenames, overlap_areas],
                                                   num_epochs=None, shuffle=True)

    l, k, a = read_pair(filename_queue)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    return _generate_image_and_label_batch(l, k, a,
                                           min_queue_examples, batch_size,
                                           shuffle=True)

def _generate_image_and_label_batch(lock_image, key_image, label, min_queue_examples,
                                    batch_size, shuffle):
    print "Image dimensions: ", lock_image.get_shape()
    num_preprocess_threads = 16
    if shuffle:
        lock_images, key_images, label_batch = tf.train.shuffle_batch(
            [lock_image, key_image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 6 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        lock_images, key_images, label_batch = tf.train.batch(
            [lock_image, key_image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 6 * batch_size)

    print "Images dimensions: ", lock_images.get_shape()

    return lock_images, key_images, tf.reshape(label_batch, [batch_size])

if __name__ == '__main__':

    locks, keys, labels = inputs(False, FLAGS.data_dir, 128)