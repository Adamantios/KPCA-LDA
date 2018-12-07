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


def _get_spam_labels() -> Labels:
    """
    Creates labels for the spambase dataset attributes.

    :return: List of strings containing the labels.
    """
    # Create an empty list.
    names = []

    # For all the word frequencies, create a label containing the word 'word-freq-', followed by its index.
    for i in range(48):
        names.append('word-freq-' + str(i))

    # For all the char frequencies, create a label containing the word 'char-freq-', followed by its index.
    for i in range(6):
        names.append('char-freq-' + str(i))

    # Create a list containing the remaining labels.
    labels = ['capital-run-length-average', 'capital-run-length-longest', 'capital-run-length-total', 'class']

    # Add remaining labels.
    for label in labels:
        names.append(label)

    return names


def load_digits(train: bool = True) -> Dataset:
    """
    Load the mnist handwritten digits dataset
    and convert the prediction labels to odd and even numbers.

    :param train: whether to load the train or the test data.
    If True, returns the train.

    If False, returns the test.

    Default value: True

    :return: Array containing the mnist handwritten digits dataset.
    """
    # Create a filename based on the train value.
    filename = 'datasets/mnist_train.csv' if train else 'datasets/mnist_test.csv'

    # Read the dataset and get its values.
    dataset = read_csv(filename, names=_get_mnist_labels()).values

    # Get x and y.
    x = dataset[:, 1:]
    y = dataset[:, 0]

    # Convert the prediction labels to odd and even numbers.
    y[y % 2 == 0] = 0
    y[y % 2 == 1] = 1

    return x, y


def database_split(db_filename: str, train_filename: str, test_filename: str) -> None:
    """
    Splits a csv dataset to 60% train and 40% test files, stratified.

    :param db_filename: the dataset's filename.
    :param train_filename: the train filename to be created.
    :param test_filename: the test filename to be created.
    """
    # Read the dataset an get its values.
    # Use string datatypes, so that we take the information as it is.
    # If floats were to be used, then the labels would be converted to floats too.
    dataset = read_csv(db_filename, dtype=numpy.str).values

    # Get x and y.
    x, y = dataset[:, :-1], dataset[:, -1]
    # Split to train and test pairs.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, stratify=y, random_state=0)

    # Concatenate the train and test pairs arrays into single arrays.
    train = numpy.column_stack((x_train, y_train))
    test = numpy.column_stack((x_test, y_test))

    # Get the spam labels.
    labels = _get_spam_labels()

    # Create Dataframes from the train and test arrays and write them to a csv file.
    DataFrame(train, columns=labels, dtype=numpy.str).to_csv(train_filename, index=False)
    DataFrame(test, columns=labels, dtype=numpy.str).to_csv(test_filename, index=False)


def load_spam(train: bool = True) -> Dataset:
    """
    Loads the spambase dataset.

    :param train: whether to load the train or the test data.
    If True, returns the train.

    If False, returns the test.

    Default value: True

    :return: Numpy representation of the dataset.
    """
    # The filename of the dataset.
    db_filename = 'datasets/spambase.csv'
    # Create Path objects using the paths were the train and test files should be.
    train_file = Path('datasets/spambase_train.csv')
    test_file = Path('datasets/spambase_test.csv')

    # If the files from the given paths do not exist, create them by splitting the spambase dataset.
    if not train_file.is_file() or not test_file.is_file():
        database_split(db_filename, train_file.absolute(), test_file.absolute())

    # Create a filename based on the train value.
    filename = train_file.absolute() if train else test_file.absolute()
    # Read the dataset.
    dataset = read_csv(filename).values
    # Get x and y.
    x, y = dataset[:, :-1], dataset[:, -1]

    return x, y


def get_email_name(class_num: int) -> str:
    """
    Gets an email's class name by its number.

    :param class_num: the number of the class name to be returned.
    :return: String containing the class name.
    """
    class_names = {
        0: 'Not Spam',
        1: 'Spam'
    }
    return class_names.get(class_num, 'Invalid')


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
