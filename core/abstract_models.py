import numpy as np
from abc import abstractmethod, ABC
from typing import Union, Tuple


class NotFittedException(Exception):
    pass


class InvalidNumOfComponentsException(Exception):
    pass


class OneSamplePassedException(Exception):
    pass


class _Model(object):
    def __init__(self):
        pass

    @staticmethod
    def _param_value(param: any) -> Union[any, str]:
        """ For param with value None, return the string auto. """
        return param if param is not None else 'auto'

    def get_params(self) -> dict:
        """
        Getter for the model's parameters and its values.

        :return: a dictionary containing the model's parameters and its values.
        """
        return {}


class _Decomposer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def _pov_to_n_components(self) -> int:
        """
        Finds the number of components needed in order to succeed the pov given.

        :return: the number of components.
        """
        pass

    @abstractmethod
    def _check_n_components(self) -> None:
        """ Checks the validity of the number of components. """
        pass

    @abstractmethod
    def _clean_eigs(self, eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Processes the eigenvalues and eigenvectors and returns them clean.

        :param eigenvalues: the eigenvalues to be cleaned.
        :param eigenvectors: the eigenvectors to be cleaned.
        :return: tuple containing the clean eigenvalues and eigenvectors.
        """
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
