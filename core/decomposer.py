import numpy as np
from abc import abstractmethod, ABC


class NotFittedException(Exception):
    pass


class _Decomposer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Method to fit the decomposer.

        :param x: the array to be fitted.
        :param y: optional array containing the class labels.
        :return: the fitted x.
        """
        pass

    @abstractmethod
    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Projects the given data to the created feature space.

        :param x: the data to be projected.
        :return: The projected data.
        """
        pass

    @abstractmethod
    def fit_transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Equivalent to fit().transform(), but more efficient, if possible.

        :param x: the data to be fitted and then transformed.
        :param y: optional array containing the class labels
        :return: the projected data.
        """
        pass

    @abstractmethod
    def get_params(self) -> dict:
        """
        Getter for the decomposer's parameters.

        :return: the decomposer's parameters.
        """
        pass
