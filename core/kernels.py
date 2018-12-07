import numpy as np
from enum import Enum, auto
from typing import NoReturn
from scipy.spatial.distance import pdist, squareform


class Kernels(Enum):
    LINEAR = auto()
    POLYNOMIAL = auto()
    RBF = auto()
    MIN = auto()


class Kernel:
    def __init__(self, kernel: Kernels = Kernels.RBF, alpha: float = None, coefficient: float = 0,
                 degree: int = 3, sigma: float = None):
        self._kernel = kernel
        self._alpha = alpha
        self._coefficient = coefficient
        self._degree = degree
        self._sigma = sigma

    @staticmethod
    def _array_dim_check(x: np.ndarray) -> NoReturn:
        """
        Checks if the passed array is 2D.
        If not, it raises a value error.
        :param x: the array to be checked.
        """
        if x.ndim != 2:
            raise ValueError('Input array should be 2 dimensional.')

    @staticmethod
    def _linear_kernel(x: np.ndarray, coefficient: float) -> np.ndarray:
        """
        Calculates a Linear Kernel matrix for the passed array.

        :param x: a 2D array.
        :param coefficient: the coefficient parameter to be used in the kernel calculation.
        :return: The calculated kernel matrix.
        """
        # Check array's dimensions.
        Kernel._array_dim_check(x)

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
    def _poly_kernel(x: np.ndarray, alpha: float, coefficient: float, degree: int) -> np.ndarray:
        """
        Calculates a Polynomial Kernel matrix for the passed array.

        :param x: a 2D array.
        :param alpha: the alpha value to be used in the kernel calculation.
        :param coefficient: coefficient: the coefficient parameter to be used in the kernel calculation.
        :param degree: the degree of the polynomial kernel.
        :return: The calculated kernel matrix.
        """
        # Check array's dimensions.
        Kernel._array_dim_check(x)

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
    def _rbf_kernel(x: np.ndarray, sigma: float) -> np.ndarray:
        """
        Calculates an Rbf Kernel matrix for the passed array.

        :param x: a 2D array.
        :param sigma: the sigma value to be used in the kernel calculation.
        :return: The calculated kernel matrix.
        """
        # Check array's dimensions.
        Kernel._array_dim_check(x)

        # Calculate the Euclidean distances for every pair of values.
        dists = pdist(x, 'sqeuclidean')

        # Convert the distances into a symmetric matrix, where xij is the distance of xi from xj.
        dists = squareform(dists)

        gamma = np.divide(1, np.multiply(2, np.square(sigma)))

        return np.exp(-gamma * dists)

    @staticmethod
    def _min_kernel(x: np.ndarray) -> np.ndarray:
        """
        Calculates a Min Kernel matrix, also known as Histogram Intersection Kernel for the passed array.

        :param x: a 2D array.
        :return: The calculated kernel matrix.
        """
        # Check array's dimensions.
        Kernel._array_dim_check(x)

        # Get the number of samples.
        n_samples = x.shape[0]

        # Initialize an array for the dot products.
        sums = np.zeros((n_samples, n_samples))

        # Calculate the dot products for every pair of sample.
        for i in range(n_samples):
            for j in range(n_samples):
                sums[i, j] = np.sum(np.minimum(x[i, :], x[j, :]))

        return sums

    def calc_array(self, x: np.ndarray) -> np.ndarray:
        """
        Calculates the kernel matrix.

        :param x: the 2D array from which the kernel will be calculated.
        :return: The calculated kernel matrix.
        """
        # Get the number of features.
        n_features = x.shape[1]

        # Set missing parameters.
        if self._alpha is None:
            self._alpha = 1 / n_features

        if self._sigma is None:
            self._sigma = np.sqrt(n_features)

        # Return the kernel matrix for the chosen kernel.
        if self._kernel == Kernels.LINEAR:
            return Kernel._linear_kernel(x, self._coefficient)
        elif self._kernel == Kernels.POLYNOMIAL:
            return Kernel._poly_kernel(x, self._alpha, self._coefficient, self._degree)
        elif self._kernel == Kernels.RBF:
            return Kernel._rbf_kernel(x, self._sigma)
        elif self._kernel == Kernels.MIN:
            return Kernel._min_kernel(x)
