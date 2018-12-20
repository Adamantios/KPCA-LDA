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

        # Sum the means of the same classes.
        for c, label in zip(range(self._n_classes), self._labels):
            means[c] += np.mean(x[y == label], axis=0, dtype=np.float64)

        return means

    def _sw(self, x: np.ndarray, y: np.ndarray, class_means: np.ndarray) -> np.ndarray:
        """
        Calculates the within class scatter matrix.

        :param x: array containing the data.
        :param y: array containing the features.
        :param class_means: array containing the mean vector of every class.
        :return: the within class scatter matrix.
        """
        # Instantiate an array for the within class scatter matrix.
        sw = np.zeros((self._n_features, self._n_features))

        # For every class label.
        for label, label_index in enumerate(self._labels):
            # Instantiate an array for the Si matrix.
            si = np.zeros((self._n_features, self._n_features))
            # Get the samples of the current class.
            grouped_samples = x[y == label]
            # Get the number of the samples of the current class.
            n_grouped_samples = grouped_samples.shape[0]

            # For every sample.
            for sample, sample_index in zip(grouped_samples, range(n_grouped_samples)):
                # Calculate the difference of the sample vector from the mean vector.
                diff = sample - class_means[label_index]
                # Make the sample's diff a column vector.
                diff = np.expand_dims(diff, axis=1)
                # Sum the dot product of the difference with itself's transpose to the Si matrix.
                si += np.dot(diff, diff.T)
            # Sum the Si results of all the classes to get Sw.
            sw += si

        return sw

    def _sb(self, class_means: np.ndarray, x_mean) -> np.ndarray:
        """
        Calculates the between class scatter matrix.

        :param class_means: array containing the mean vector of every class.
        :return: the between class scatter matrix.
        """
        # Instantiate an array for the between class scatter matrix.
        sb = np.zeros((self._n_features, self._n_features))

        # For every class mean vector.
        for mean_vec, count in zip(class_means, self._labels_counts):
            # Convert mean vector to a column vector.
            mean_vec_column = np.expand_dims(mean_vec, axis=1)
            # Get the difference of the current mean vector with the overall mean vector.
            diff = mean_vec_column - x_mean
            # Multiply the number of the class instances
            # with the dot product of the difference with itself's transpose
            # and sum the result to the Sb matrix.
            sb += count * np.dot(diff, diff.T)

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

        # Get the between class scatter matrix array. Pass x_mean as a column vector.
        sb = self._sb(class_means, np.expand_dims(x_mean, axis=1))

        # Calculate the product of the sw's inverse and sb.
        sw_inv_sb = np.dot(np.linalg.inv(sw), sb)

        # Get the eigenvalues and eigenvectors of the sw-1*sb, in ascending order.
        eigenvalues, eigenvectors = np.linalg.eigh(sw_inv_sb)

        # TODO calc explained var.

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
