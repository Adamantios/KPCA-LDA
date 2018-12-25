from os.path import dirname, abspath, join

"""------------------------- Can be modified. ------------------------"""

# Whether plots should be created or not.
CREATE_PLOTS = True

# Whether plots should be saved or not.
SAVE_PLOTS = False

# Whether predictions should be saved as an excel file or not.
SAVE_PRED_RESULTS = False

# The output directory's name.
OUT_DIR_NAME = 'out'
# The datasets folder name.
DATASETS = 'datasets'

# The seizure detection dataset's filename.
DATASET_SEIZURE = 'epileptic_seizure.csv'
# The seizure detection train dataset's filename.
DATASET_SEIZURE_TRAIN = 'epileptic_seizure_train.csv'
# The seizure detection test dataset's filename.
DATASET_SEIZURE_TEST = 'epileptic_seizure_test.csv'

# The genes data filename.
DATA_GENES = 'genes.csv'
# The genes labels filename.
LABELS_GENES = 'genes_labels.csv'
# The genes train dataset's filename.
DATASET_GENES_TRAIN = 'genes_train.csv'
# The genes test dataset's filename.
DATASET_GENES_TEST = 'genes_test.csv'

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

# The path to the genes data.
__GENES_DATA_PATH = join(__ROOT, DATASETS, DATA_GENES)
# The path to the genes labels.
__GENES_LABELS_PATH = join(__ROOT, DATASETS, LABELS_GENES)
# The path to the genes train dataset.
__GENES_TRAIN_PATH = join(__ROOT, DATASETS, DATASET_GENES_TRAIN)
# The path to the genes test dataset.
__GENES_TEST_PATH = join(__ROOT, DATASETS, DATASET_GENES_TEST)
