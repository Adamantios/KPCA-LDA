import numpy
from pathlib import Path
from typing import Tuple, List
from pandas import read_csv, DataFrame
from sklearn.model_selection import train_test_split

Labels = List[str]
Dataset = Tuple[numpy.ndarray, numpy.ndarray]


def _get_mnist_labels() -> Labels:
    """
    Creates labels for the mnist dataset attributes.

    :return: List of strings containing the labels.
    """
    # Create a list with the prediction label's name.
    names = ['number']

    # For every pixel, create a label containing the word 'pixel', followed by its index.
    for i in range(784):
        names.append('pixel' + str(i))

    return names


def load_digits(train: bool = True) -> Dataset:
    """
    Load the original mnist handwritten digits dataset.

    :param train: whether to load the train or the test data.
    If True, returns the train.

    If False, returns the test.

    Default value: True

    :return: Array containing the mnist handwritten digits dataset.
    """
    # Create a filename based on the train value.
    filename = 'datasets/mnist_train.csv' if train else 'datasets/mnist_test.csv'

    # Read the dataset and get its values.
    dataset = read_csv(filename, names=_get_mnist_labels(), nrows=2500).values

    # Get x and y.
    x = dataset[:, 1:]
    y = dataset[:, 0]

    return x, y


# def database_split(db_filename: str, train_filename: str, test_filename: str) -> None:
#     """
#     Splits a csv dataset to 60% train and 40% test files, stratified.
#
#     :param db_filename: the dataset's filename.
#     :param train_filename: the train filename to be created.
#     :param test_filename: the test filename to be created.
#     """
#     # Read the dataset an get its values.
#     # Use string datatypes, so that we take the information as it is.
#     # If floats were to be used, then the labels would be converted to floats too.
#     dataset = read_csv(db_filename, dtype=numpy.str).values
#
#     # Get x and y.
#     x, y = dataset[:, :-1], dataset[:, -1]
#     # Split to train and test pairs.
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, stratify=y, random_state=0)
#
#     # Concatenate the train and test pairs arrays into single arrays.
#     train = numpy.column_stack((x_train, y_train))
#     test = numpy.column_stack((x_test, y_test))
#
#     # Get the spam labels.
#     labels = _get_spam_labels()
#
#     # Create Dataframes from the train and test arrays and write them to a csv file.
#     DataFrame(train, columns=labels, dtype=numpy.str).to_csv(train_filename, index=False)
#     DataFrame(test, columns=labels, dtype=numpy.str).to_csv(test_filename, index=False)


def get_digit_name(class_num: int) -> str:
    """
    Gets a digit's class name by its number.

    :param class_num: the number of the class name to be returned.
    :return: String containing the class name.
    """
    class_names = {
        0: 'Even',
        1: 'Odd'
    }
    return class_names.get(class_num, 'Invalid')
