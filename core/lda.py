import numpy as np
from typing import Tuple
from core.decomposer import _Decomposer


class Lda(_Decomposer):
    def __init__(self):
        super().__init__()

        self._labels = None
        self._labels_counts = None

    def _n_classes(self):
        """
        Calculates the number of the classes.
        :return: The number of the classes.
        """
        return len(self._labels)

    def _class_means(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculates the means for every class.

        :param x: array containing the data.
        :param y: array containing the features.
        :return: array containing the means.
        """
        # Get the number of classes and the number of features.
        n_features = x.shape[1]
        # Instantiate an array to hold the mean of every feature for each class.
        means = np.zeros((self._n_classes(), n_features))

        # Sum the means of the features of the same classes.
        for c, label in zip(range(self._n_classes()), self._labels):
            for f in range(n_features):
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
        sb = np.zeros((self._n_classes(), self._n_classes()))

        # Calculate between class scatter matrix
        for count in self._labels_counts:
            sb += np.multiply(count, np.dot(means_diff, means_diff.T))

        return sb

    def _sw(self, x: np.ndarray, y: np.ndarray, class_means: np.ndarray) -> np.ndarray:
        """
        Calculates the within class scatter matrix.

        :return: the within class scatter matrix.
        """
        # Instantiate an array for the within class scatter matrix.
        sw = np.zeros((self._n_classes(), self._n_classes()))

        feature_mean_dist = np.zeros(x.shape[1])

        # Calculate within class scatter matrix
        for label in self._labels:
            for feature in x[y == label]:
                feature_mean_dist[feature] += feature - class_means[label, feature]

            sw += np.dot(feature_mean_dist, feature_mean_dist.T)

        return sw

    def fit(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Get the labels and the number of instances for every class.
        self._labels, self._labels_counts = np.unique(y, return_counts=True)

        # Calculate the x mean using float64 to get a more accurate result.
        x_mean = x.mean(dtype=np.float64)

        # Get the class means of every feature.
        class_means = self._class_means(x, y)

        # Get the between class scatter matrix array.
        sb = self._sb(class_means - x_mean)

        # Get the within class scatter matrix array.
        sw = self._sw(x, y, class_means)

        return sb, sw

    def transform(self, x: np.ndarray) -> np.ndarray:
        pass

    def fit_transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        pass

    def get_params(self) -> dict:
        pass
