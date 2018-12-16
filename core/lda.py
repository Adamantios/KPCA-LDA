import numpy as np
from core.decomposer import _Decomposer, NotFittedException


class Lda(_Decomposer):
    def __init__(self, remove_zeros=True):
        super().__init__()

        self.remove_zeros = remove_zeros
        self._labels = None
        self._labels_counts = None
        self._n_classes = None
        self._n_features = None
        self._w = None

    @staticmethod
    def _check_if_possible(x: np.ndarray) -> None:
        """
        Checks if it possible to perform LDA.

        If not, raises an exception.

        :param x: the passed data.
        """
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
        Calculates the mean of every feature for each class.

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

    def _sw(self, x: np.ndarray, y: np.ndarray, class_means: np.ndarray) -> np.ndarray:
        """
        Calculates the within class scatter matrix.

        :return: the within class scatter matrix.
        """
        # Instantiate an array for the within class scatter matrix.
        sw = np.zeros((self._n_features, self._n_features))

        # Calculate within class scatter matrix
        for label, label_index in enumerate(self._labels):
            # Instantiate an array for the Si matrix.
            si = np.zeros((self._n_features, self._n_features))
            grouping_mask = y == label
            grouped_samples = x[grouping_mask]
            n_grouped_samples = grouped_samples.shape[0]
            diffs = np.zeros((n_grouped_samples, self._n_features))

            # Calculate for every class the difference of each sample's feature from the mean feature.
            for sample, sample_index in zip(grouped_samples, range(n_grouped_samples)):
                for feature in range(self._n_features):
                    diffs[sample_index, feature] = sample[feature] - class_means[label_index, feature]

                sample_diff_2d = np.expand_dims(diffs[sample_index], axis=1)
                si += np.dot(sample_diff_2d, sample_diff_2d.T)
            sw += si

        return sw

    def _sb(self, x, y, class_means: np.ndarray, x_mean) -> np.ndarray:
        """
        Calculates the between class scatter matrix.

        :param means_diff: the differences between the mean classes and the total classes mean.
        :return: The between class scatter matrix.
        """
        # Instantiate an array for the between class scatter matrix.
        sb = np.zeros((self._n_features, self._n_features))

        for label, mean_vec, count in zip(self._labels, class_means, self._labels_counts):
            mean_vec = mean_vec.reshape(self._n_features, 1)  # make column vector
            x_mean = x_mean.reshape(self._n_features, 1)  # make column vector
            sb += count * (mean_vec - x_mean).dot((mean_vec - x_mean).T)

        return sb

    def fit(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fits the Lda model with a given dataset.

        :param x: the data to be fitted.
        :param y: the class labels.
        :return: an array of the eigenvectors created.
        """
        self._check_if_possible(x)
        self.__set_state(x, y)

        # Calculate the x mean using float64 to get a more accurate result.
        x_mean = x.mean(axis=0, dtype=np.float64)

        # Get the class means of every feature.
        class_means = self._class_means(x, y)

        # Get the within class scatter matrix array.
        sw = self._sw(x, y, class_means)

        # Get the between class scatter matrix array.
        sb = self._sb(x, y, class_means, x_mean)

        # Calculate the product of the sw's inverse and sb.
        sw_inv_sb = np.dot(np.linalg.inv(sw), sb)

        # Get the eigenvalues and eigenvectors of the sw-1*sb, in ascending order.
        eigenvalues, eigenvectors = np.linalg.eigh(sw_inv_sb)

        # If user has chosen to remove the eigenvectors which have zero eigenvalues.
        if self.remove_zeros:
            # Get the indexes of the zero eigenvalues.
            unwanted_indexes = np.where(np.isclose(eigenvalues, 0))

            # Get all eigenvectors which have zero eigenvalues.
            eigenvectors = np.delete(eigenvectors, unwanted_indexes, axis=1)

        # Sort the eigenvalues and eigenvectors in descending order.
        self._w = np.flip(eigenvectors, axis=1)

        self._w = np.delete(eigenvectors, np.s_[self._n_classes - 1:], axis=1)

        return self._w

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Project the data to the new dimension.

        :param x: the data to be transformed.
        :return: the projected data.
        """
        # If LDA has not been fitted yet, raise an Exception.
        if self._w is None:
            raise NotFittedException('KPCA has not been fitted yet!')

        return np.dot(x, self._w)

    def fit_transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Equivalent to fit().transform().

        :param x: the data to be fitted and then transformed.
        :param y: the class labels.
        :return: the transformed data.
        """
        self.fit(x, y)
        return self.transform(x)

    def get_params(self) -> dict:
        pass
