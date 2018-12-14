from os.path import dirname, abspath, join

"""------------------------- Can be modified. ------------------------"""

# The output directory's name.
OUT_DIR_NAME = 'out'
# The datasets folder name.
DATASETS = 'datasets'
# The seizure detection dataset's name.
DATASET_SEIZURE = 'epileptic_seizure.csv'
# The seizure detection train dataset's name.
DATASET_SEIZURE_TRAIN = 'epileptic_seizure_train.csv'
# The seizure detection test dataset's name.
DATASET_SEIZURE_TEST = 'epileptic_seizure_test.csv'
# The digits recognition train dataset's name.
DATASET_DIGITS_TRAIN = 'mnist_train.csv'
# The digits recognition test dataset's name.
DATASET_DIGITS_TEST = 'mnist_test.csv'

""" ----------------------------------------------------------------- """

""" ------------------------- Do not modify. -------------------------"""

# The root folder.
__ROOT = dirname(abspath(__file__))
# The output folder's path.
__OUT_PATH = join(__ROOT, OUT_DIR_NAME)
# The path to the seizure detection dataset.
__SEIZURE_PATH = join(__ROOT, DATASETS, DATASET_SEIZURE)
# The path to the seizure detection train dataset.
__SEIZURE_TRAIN_PATH = join(__ROOT, DATASETS, DATASET_SEIZURE_TRAIN)
# The path to the seizure detection test dataset.
__SEIZURE_TEST_PATH = join(__ROOT, DATASETS, DATASET_SEIZURE_TEST)
# The path to the digits recognition train dataset.
__DIGITS_TRAIN_PATH = join(__ROOT, DATASETS, DATASET_DIGITS_TRAIN)
# The path to the digits recognition train dataset.
__DIGITS_TEST_PATH = join(__ROOT, DATASETS, DATASET_DIGITS_TEST)
