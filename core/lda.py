import numpy as np
from typing import Tuple
from core.decomposer import _Decomposer


class Lda(_Decomposer):
    def __init__(self):
        super().__init__()

        self._labels = None
        self._labels_counts = None
        self._n_classes = None
        self._n_features = None
        self._w = None

    @staticmethod
    def _check_if_possible(x: np.ndarray) -> None:
        if x.shape[0] < 2:
            raise ValueError('Cannot perform Lda for 1 sample.')

    def __set_state(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Sets the object's state.

        :param x: array containing the samples.
        :param y: array containing the labels.
        """
        # Get the labels and the number of instances for every class.
        self._labels, self._labels_counts = np.unique(y, return_counts=True)
        # Get the number of classes.
        self._n_classes = len(self._labels)
        # Get the number of features.
        self._n_features = x.shape[1]

    def _class_means(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculates the means for every class.

        :param x: array containing the data.
        :param y: array containing the features.
        :return: array containing the means.
        """
        # Instantiate an array to hold the mean of every feature for each class.
        means = np.zeros((self._n_classes, self._n_features))

        # Sum the means of the features of the same classes.
        for c, label in zip(range(self._n_classes), self._labels):
            for f in range(self._n_features):
                means[c, f] += np.sum(np.mean(x[y == label, f]))

        return means

    def _sb(self, means_diff: np.ndarray) -> np.ndarray:
        """
        Calculates the between class scatter matrix.

        :param means_diff: the differences between
        the means of the features of the same classes and the total mean.
        :return: The between class scatter matrix.
        """
        # Instantiate an array for the between class scatter matrix.
        sb = np.zeros((self._n_features, self._n_features))

        # Calculate between class scatter matrix
        for count in self._labels_counts:
            sb += np.multiply(count, np.dot(means_diff.T, means_diff))

        return sb

    def _sw(self, x: np.ndarray, y: np.ndarray, class_means: np.ndarray) -> np.ndarray:
        """
        Calculates the within class scatter matrix.

        :return: the within class scatter matrix.
        """
        # Instantiate an array for the within class scatter matrix.
        sw = np.zeros((self._n_features, self._n_features))

        # Calculate within class scatter matrix
        for label, label_index in zip(self._labels, range(self._n_classes)):
            # Calculate for every class(label param) the distance of each feature from its mean.
            features_means_dists = x[y == label] - class_means[label_index]
            sw += np.dot(features_means_dists.T, features_means_dists)

        return sw

    def fit(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self._check_if_possible(x)
        self.__set_state(x, y)

        # Calculate the x mean using float64 to get a more accurate result.
        x_mean = x.mean(dtype=np.float64)

        # Get the class means of every feature.
        class_means = self._class_means(x, y)

        # Get the between class scatter matrix array.
        sb = self._sb(class_means - x_mean)

        # Get the within class scatter matrix array.
        sw = self._sw(x, y, class_means)

        # Calculate the product of the sw's inverse and sb.
        sw_inv_sb = np.dot(np.linalg.inv(sw), sb)

        # Get the eigenvalues and eigenvectors of the sw-1*sb, in ascending order.
        eigenvalues, eigenvectors = np.linalg.eigh(sw_inv_sb)

        # Get the indexes of the negative or zero eigenvalues.
        unwanted_indexes = np.where(eigenvalues <= 0)

        # Get all the non negative or zero eigenvectors.
        self._w = np.delete(eigenvectors, unwanted_indexes, axis=1)

        # Sort the eigenvalues and eigenvectors in descending order.
        self._w = np.flip(self._w, axis=1)

        return self._w

    def transform(self, x: np.ndarray) -> np.ndarray:
        return np.dot(x, self._w)

    def fit_transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(x, y)
        return self.transform(x)

    def get_params(self) -> dict:
        pass
