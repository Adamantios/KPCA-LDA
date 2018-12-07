import logging
import logging.handlers
from datetime import datetime
from os import path, makedirs


def create_folder(folder_name: str) -> None:
    """Creates a folder in the project root.

    :param folder_name: the name of the folder to be created.
    """
    if not path.exists(folder_name):
        makedirs(folder_name)


class Logger:
    def __init__(self, name: str = 'ResultsLogger', folder: str = 'logs', filename: str = 'results',
                 extension: str = 'log'):
        # Create folder if it does not exist.
        create_folder(folder)

        # Set up a logger.
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.INFO)

        # Add the log message handler to the logger.
        handler = logging.handlers.RotatingFileHandler(folder + '/' + filename + '.' + extension)
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
