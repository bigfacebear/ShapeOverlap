import os

# Number of images to process in a batch.
batch_size = 128

local = True

if local:
    dataset_dir = './Dataset'
    train_dir = './ShapeOverlap_train'
else:
    dataset_dir = '/cstor/xsede/users/xs-qczhao/Dataset'
    train_dir = '/cstor/xsede/users/xs-qczhao/ShapeOverlap_train'

# dataset_name = 'filled_dataset_notrans_150x150_100000'
dataset_name = 'filled_dataset_100000'
# dataset_name = 'area_dataset'

data_dir = os.path.join(dataset_dir, dataset_name)

# Number of batches to run.
max_steps = 100000 #1000000
# Whether to log device placement.
log_device_placement = False
# How often to log results to the console.
log_frequency = 1000

# Global constants describing the MSHAPES data set.
IMAGE_SIZE = 200
NUM_CLASSES = 1
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 80000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 20000

# Constants describing the training process.
NUM_EPOCHS_PER_DECAY = 5.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.5  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

# Where to download the MSHAPES dataset from
DATA_URL = 'https://electronneutrino.com/affinity/overlap/dataset.zip'
# DATA_URL = 'https://pandownload.zju.edu.cn/download/93c5378f642e4e499625d6f84b5b7f9d/76bb50a3fa9fb5671e6c0880ad14e1dc05878c5da6f2b6e7fd3136391d91bb8d/filled_dataset_notrans_150x150_100000.tar.gz'
# DATA_URL = 'https://pandownload.zju.edu.cn/download/ea840a63bd334ae9a455288b92f3e240/1c6d94d225ed1112c0b05105c6f7f775f388faf2c8ab471766c478df5a3ab2ad/area_dataset.tar.gz'

NOTIFICATION_EMAIL = 'qc_zhao@outlook.com'

CHECK_DATASET = True

RESTORE = False
RESTORE_FROM = 'summaries/netstate/saved_state-3000'

# Train the model using fp16.
use_fp16 = False
