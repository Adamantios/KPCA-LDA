import numpy as np
from enum import Enum, auto
from core.abstract_models import _Model
from scipy.spatial.distance import pdist, squareform, cdist


class Kernels(Enum):
    LINEAR = auto()
    POLYNOMIAL = auto()
    RBF = auto()
    MIN = auto()


class InvalidDimensionsException(Exception):
    pass


class InvalidKernelException(Exception):
    pass


class IncompatibleShapesException(Exception):
    pass


class Kernel(_Model):
    def __init__(self, kernel: Kernels = Kernels.RBF, alpha: float = None, coefficient: float = 0, degree: int = 3,
                 sigma: float = None):
        super().__init__()
        self._kernel = kernel
        self._alpha = alpha
        self._coefficient = coefficient
        self._degree = degree
        self._sigma = sigma

    @staticmethod
    def _array_dim_check(x: np.ndarray) -> None:
        """
        Checks if the passed array is 2D.
        If not, it raises an InvalidDimensionsException.

        :param x: the array to be checked.
        """
        if x.ndim == 1:
            raise InvalidDimensionsException('Input arrays should be 2 dimensional.\n'
                                             'Got {} instead.\n'
                                             'You could use np.expand_dims(array, axis=0), '
                                             'in order to convert a 1D array with one sample to 2D\n'
                                             'or np.expand_dims(array, axis=1), '
                                             'in order to convert a 1D array with one feature to 2D.'
                                             .format(x.shape))
        if x.ndim != 2:
            raise InvalidDimensionsException('Input arrays should be 2 dimensional.\n'
                                             'Got {} instead.'
                                             .format(x.shape))

    @staticmethod
    def _arrays_check(x: np.ndarray, y: np.ndarray) -> None:
        """
        Checks the passed arrays dimensions.
        If y is passed too, then check if the column size of the two arrays is the same.

        :param x: the x array to be checked.
        :param y: the y array to be checked.
        """
        Kernel._array_dim_check(x)

        if y is not None:
            Kernel._array_dim_check(y)
            if x.shape[1] != y.shape[1]:
                raise IncompatibleShapesException(
                    'Arrays column size should be the same. Got {} and {} instead'.format(x.shape[1], y.shape[1]))

    @staticmethod
    def _linear_kernel(x: np.ndarray, y: np.ndarray, coefficient: float) -> np.ndarray:
        """
        Calculates the linear kernel matrix between x and y or for x if y is not passed.

        :param x: the 2D array from which the kernel will be calculated.
        :param y: the optional 1D or 2D array with which the kernel will be calculated between x.
        :return: The calculated kernel matrix.
        """
        if y is not None:
            # Calculate the dot products.
            dots = np.dot(x, y.T)

        else:
            # Get the number of samples.
            n_samples = x.shape[0]

            # Initialize an array for the dot products.
            dots = np.zeros((n_samples, n_samples))

            # Calculate the dot products for every pair of sample.
            for i in range(n_samples):
                for j in range(n_samples):
                    dots[i, j] = np.dot(x[i, :].T, x[j, :])

        # Add coefficient before returning the kernel array.
        return dots + coefficient

    @staticmethod
    def _poly_kernel(x: np.ndarray, y: np.ndarray, alpha: float, coefficient: float, degree: int) -> np.ndarray:
        """
        Calculates the polynomial kernel matrix between x and y or for x if y is not passed.

        :param x: the 2D array from which the kernel will be calculated.
        :param y: the optional 1D or 2D array with which the kernel will be calculated between x.
        :param alpha: the alpha value to be used in the kernel calculation.
        :param coefficient: coefficient: the coefficient parameter to be used in the kernel calculation.
        :param degree: the degree of the polynomial kernel.
        :return: The calculated kernel matrix.
        """
        if y is not None:
            # Calculate the dot products.
            dots = np.dot(x, y.T)

        else:
            # Get the number of samples.
            n_samples = x.shape[0]

            # Initialize an array for the dot products.
            dots = np.zeros((n_samples, n_samples))

            # Calculate the dot products for every pair of sample.
            for i in range(n_samples):
                for j in range(n_samples):
                    dots[i, j] = np.dot(x[i, :].T, x[j, :])

        # Multiply with alpha, add coefficient
        # and raise the result in the power of degree before returning the kernel array.
        return np.power(alpha * dots + coefficient, degree)

    @staticmethod
    def _rbf_kernel(x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
        """
        Calculates the rbf kernel matrix between x and y or for x if y is not passed.

        :param x: the 2D array from which the kernel will be calculated.
        :param y: the optional 1D or 2D array with which the kernel will be calculated between x.
        :param sigma: the sigma value to be used in the kernel calculation.
        :return: The calculated kernel matrix.
        """
        if y is not None:
            dists = cdist(x, y, 'sqeuclidean')

        else:
            # Calculate the Euclidean distances for every pair of values.
            dists = pdist(x, 'sqeuclidean')
            # Convert the distances into a symmetric matrix, where xij is the distance of xi from xj.
            dists = squareform(dists)

        # Calculate gamma.
        gamma = np.divide(1, np.multiply(2, np.square(sigma)))

        return np.exp(-gamma * dists)

    @staticmethod
    def _min_kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculates the Min Kernel matrix, also known as Histogram Intersection Kernel,
        between x and y or for x if y is not passed.

        :param x: the 2D array from which the kernel will be calculated.
        :param y: the optional 1D or 2D array with which the kernel will be calculated between x.
        :param x: a 2D array.
        :return: The calculated kernel matrix.
        """
        # Get the x's number of samples.
        x_n_samples = x.shape[0]

        if y is not None:
            # Get the y's number of samples.
            y_n_samples = y.shape[0]

            # Initialize an array for the sums.
            sums = np.zeros((y_n_samples, x_n_samples))
            for i in range(y_n_samples):
                for j in range(x_n_samples):
                    sums[i, j] = np.sum(np.minimum(x[i, :], y[j, :]))

        else:
            # Get the number of samples.
            x_n_samples = x_n_samples

            # Initialize an array for the sums.
            sums = np.zeros((x_n_samples, x_n_samples))

            # Calculate the sums for every pair of sample.
            for i in range(x_n_samples):
                for j in range(x_n_samples):
                    sums[i, j] = np.sum(np.minimum(x[i, :], x[j, :]))

        return sums

    def calc_matrix(self, x: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """
        Calculates the kernel matrix between x and y or for x if y is not passed.

        :param x: the 2D array from which the kernel will be calculated.
        :param y: the optional 1D or 2D array with which the kernel will be calculated between x.
        :return: The calculated kernel matrix.
        """
        # Check arrays dimensions.
        Kernel._arrays_check(x, y)

        # Get the number of features.
        n_features = x.shape[1]

        # Set missing parameters.
        if self._alpha is None:
            self._alpha = 1 / n_features

        if self._sigma is None:
            self._sigma = np.sqrt(n_features / 2)

        # Return the kernel matrix for the chosen kernel.
        if self._kernel == Kernels.LINEAR:
            return Kernel._linear_kernel(x, y, self._coefficient)
        elif self._kernel == Kernels.POLYNOMIAL:
            return Kernel._poly_kernel(x, y, self._alpha, self._coefficient, self._degree)
        elif self._kernel == Kernels.RBF:
            return Kernel._rbf_kernel(x, y, self._sigma)
        elif self._kernel == Kernels.MIN:
            return Kernel._min_kernel(x, y)
        else:
            raise InvalidKernelException('Please choose a valid Kernel method.')

    def get_params(self) -> dict:
        """
        Getter for the kernel's parameters.

        :return: the kernel's parameters.
        """
        # Create params dictionary.
        params = dict(kernel=self._param_value(self._kernel.name),
                      alpha=self._param_value(self._alpha),
                      coefficient=self._param_value(self._coefficient),
                      degree=self._param_value(self._degree),
                      sigma=self._param_value(self._sigma))

        return params
