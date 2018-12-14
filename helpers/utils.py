import logging
import logging.handlers
import numpy as np
import pandas
from definitions import __OUT_PATH
from datetime import datetime
from os import path, makedirs


def create_folder(folder_name: str) -> str:
    """Creates a folder in an oyt folder under root.

    :param folder_name: the name of the folder to be created.
    """
    # Create out path if needed.
    if not path.exists(__OUT_PATH):
        makedirs(__OUT_PATH)

    # Set the folder's path.
    folder_path = path.join(__OUT_PATH, folder_name)
    if not path.exists(folder_path):
        makedirs(folder_path)

    return folder_path


def create_excel(data: dict = None, folder: str = 'excels', filename: str = 'excel', extension: str = 'xlsx',
                 sheet_name: str = 'sheet1') -> None:
    """
    Creates an excel file from a dictionary.

    :param data: the data contained in a dict.
    :param folder: the folder under which the file is going to be created.
    :param filename: the excel's filename.
    :param extension: the file's extension.
    :param sheet_name: the excel's sheet name.
    """
    # Create folder for the file, if it doesn't exist..
    folder_path = create_folder(folder)

    # Add extension to filename.
    filename += '.' + extension

    # Set the filepath.
    filepath = path.join(folder_path, filename)

    # Create a dataframe from the passed data and set decimals to 4.
    dataframe = pandas.DataFrame(data).round(4)

    # Dump dataframe to excel file.
    dataframe.to_excel(filepath, sheet_name=sheet_name, index=False, engine='xlsxwriter')


def cm_to_accuracies(cm: np.ndarray) -> np.ndarray:
    """
    Convert confusion matrix to an array with accuracy for each class.
    :param cm: the confusion matrix.
    :return: Accuracy for each class
    """
    # Divide confusion matrix row's values with their sums.
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # Return the diagonal, which is the accuracy of each class.
    return cm.diagonal()


class Logger:
    def __init__(self, name: str = 'ResultsLogger', folder: str = 'logs', filename: str = 'results',
                 extension: str = 'log'):
        # Create folder if it does not exist.
        folder_path = create_folder(folder)

        # Set up a logger.
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.INFO)

        # Add the log message handler to the logger.
        handler = logging.handlers.RotatingFileHandler(path.join(folder_path, filename) + '.' + extension)
        self._logger.addHandler(handler)

        self._add_header()

    def _add_header(self) -> None:
        """Logs a header containing the current datetime."""
        self.log('----------- ' + str(datetime.now()) + ' -----------\n')

    def _add_separator(self) -> None:
        """Appends a separator to the logfile, for the sessions to be separated easily."""
        self._logger.info('\n\n-----------------------------------------------------------------------------------\n\n')

    def log(self, message: str, print_message: bool = True) -> None:
        """
        Logs a message.

        :param message: the message to be logged.
        :param print_message: whether the message will also be print or not.
        """
        self._logger.info(message)

        if print_message:
            print(message)

    def close(self) -> None:
        """Closes the logger."""
        self._add_separator()
