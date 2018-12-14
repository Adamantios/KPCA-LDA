import numpy as np

from core.decomposer import _Decomposer


class Lda(_Decomposer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _class_means(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculates the means for every class.

        :param x: array containing the data.
        :param y: array containing the features.
        :return: array containing the means.
        """
        # Get the labels.
        labels = np.unique(y)
        # Get the number of classes and the number of features.
        n_classes, n_features = len(labels), x.shape[1]
        # Instantiate an array to hold the mean of every feature for each class.
        means = np.zeros((n_classes, n_features))

        # Sum the means of the features of the same classes.
        for c, label in zip(range(n_classes), labels):
            for f in range(n_features):
                means[c, f] += np.sum(np.mean(x[y == label, f]))

        return means

    @staticmethod
    def _sb(y, means_diff: np.ndarray) -> np.ndarray:
        # Get the labels and the number of instances for every class.
        labels, counts = np.unique(y, return_counts=True)
        # Get the number of classes.
        n_classes = len(labels)

        # Instantiate an array for the between class scatter matrix.
        sb = np.zeros((n_classes, n_classes))

        # Calculate between class scatter matrix
        for index, count in zip(range(n_classes), counts):
            sb += np.multiply(count, np.dot(means_diff, means_diff.T))

        return sb

    def fit(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # Calculate the x mean using float64 to get more a accurate result.
        x_mean = x.mean(dtype=np.float64)

        class_means = Lda._class_means(x, y)
        means_diff = class_means - x_mean

        sb = self._sb(y, means_diff)

        return sb

    def transform(self, x: np.ndarray) -> np.ndarray:
        pass

    def fit_transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        pass

    def get_params(self) -> dict:
        pass
