import numpy as np
from typing import Union, Tuple
from core.abstract_models import _Decomposer, NotFittedException, InvalidNumOfComponentsException, \
    OneSamplePassedException, _Model


class LdaNotFeasibleException(Exception):
    pass


class Lda(_Model, _Decomposer):
    def __init__(self, n_components: Union[int, float] = None, remove_zeros: bool = True):
        super().__init__()

        self.n_components = n_components
        self.remove_zeros = remove_zeros
        self.explained_var = None
        self._labels = None
        self._labels_counts = None
        self._n_classes = None
        self._n_features = None
        self._w = None

    def _check_if_possible(self, x: np.ndarray) -> None:
        """
        Checks if it possible to perform LDA.

        If not, raises an exception.

        :param x: the passed data.
        """
        n_samples = x.shape[0]

        if n_samples < 2:
            raise OneSamplePassedException('Cannot perform Lda for 1 sample.')

        if n_samples < self._n_features:
            raise LdaNotFeasibleException('Lda is not feasible, '
                                          'if the number of components is less than the number of features.'
                                          'You seem to have {} components and {} features.'
                                          .format(n_samples, self._n_features))

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

    def _pov_to_n_components(self) -> int:
        """
        Finds the number of components needed in order to succeed the pov given.

        :return: the number of components.
        """
        # Get the proportion of variance.
        pov = np.cumsum(self.explained_var)

        # Get the index of the nearest pov value with the given pov preference.
        nearest_value_index = (np.abs(pov - self.n_components)).argmin()

        return nearest_value_index + 1

    def _check_n_components(self) -> None:
        """
        Adds a value to n components if needed.

        If user has not given a value, set it with the number of features minus 1.

        If the number passed is bigger than the number of features, set it with the number of features.

        If proportion of variance has been given,
        calculate the number of features which give the closest pov possible.

        If an invalid number has been passed, raise an exception.
        """
        # If num of components has not been passed, return number of classes - 1.
        if self.n_components is None:
            self.n_components = self._n_classes - 1
        # If n_components passed is bigger than or equal with the number of classes, use number of classes - 1.
        elif self.n_components >= self._n_classes:
            self.n_components = self._n_classes - 1
        # If n components have been given a correct value, pass
        elif 1 <= self.n_components < self._n_classes:
            pass
        # If pov has been passed, return as many n_components as needed.
        elif 0 < self.n_components < 1:
            self.n_components = self._pov_to_n_components()
        # Otherwise raise exception.
        else:
            raise InvalidNumOfComponentsException('The number of components should be between 1 and {}, '
                                                  'or between (0, 1) for the pov, '
                                                  'in order to choose the number of components automatically.\n'
                                                  'Got {} instead.'
                                                  .format(self._n_classes, self.n_components))

        # Keep explained var for n components only.
        self.explained_var = self.explained_var[:self.n_components]

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
        for label_index, label in enumerate(self._labels):
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

    def _sb(self, class_means: np.ndarray, x_mean: np.ndarray) -> np.ndarray:
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
        self.__set_state(x, y)
        self._check_if_possible(x)

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
        # Process the eigenvalues and eigenvectors.
        eigenvalues, eigenvectors = self._clean_eigs(eigenvalues, eigenvectors)

        # Calculate the explained variance.
        self.explained_var = np.divide(eigenvalues, np.sum(eigenvalues))
        # Correct the number of components if needed.
        self._check_n_components()

        # Store only the needed eigenvectors.
        self._w = np.delete(eigenvectors, np.s_[self.n_components:], axis=1)

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
        """
        Getter for the lda's parameters.

        :return: the lda's parameters.
        """
        # Create params dictionary.
        params = dict(n_components=self._param_value(self.n_components),
                      remove_zeros=self.remove_zeros)

        return params

    def _clean_eigs(self, eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Processes the eigenvalues and eigenvectors and returns them clean.

        :param eigenvalues: the eigenvalues to be cleaned.
        :param eigenvectors: the eigenvectors to be cleaned.
        :return: tuple containing the clean eigenvalues and eigenvectors.
        """
        # Get the indexes of the negative eigenvalues.
        unwanted_indexes = np.where(eigenvalues < 0)
        # Get all the eigenvalues which are not negative and eigenvectors corresponding to them.
        eigenvalues = np.delete(eigenvalues, unwanted_indexes)
        eigenvectors = np.delete(eigenvectors, unwanted_indexes, axis=1)

        # If user has chosen to remove the eigenvectors which have zero eigenvalues.
        if self.remove_zeros:
            # Get the indexes of the zero eigenvalues.
            unwanted_indexes = np.where(np.isclose(eigenvalues, 0))
            # Get all eigenvectors which do not have zero eigenvalues.
            eigenvectors = np.delete(eigenvectors, unwanted_indexes, axis=1)

        # Sort the eigenvectors and the eigenvalues in descending order.
        eigenvectors = np.flip(eigenvectors, axis=1)
        eigenvalues = np.flip(eigenvalues)

        return eigenvalues, eigenvectors
