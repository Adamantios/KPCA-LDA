import numpy as np
from enum import Enum, auto
from typing import NoReturn
from scipy.spatial.distance import pdist, squareform


class Kernel(Enum):
    LINEAR = auto()
    POLYNOMIAL = auto()
    RBF = auto()


def _array_dim_check(x: np.ndarray) -> NoReturn:
    if x.ndim != 2:
        raise AttributeError('Input array should be 2 dimensional.')


def _linear_kernel(x: np.ndarray, coefficient: float) -> np.ndarray:
    # Check array's dimensions.
    _array_dim_check(x)

    # Get the number of features.
    n_samples = x.shape[0]

    # Initialize an array for the dot products.
    dots = np.zeros((n_samples, n_samples))

    # Calculate the dot products for every pair of sample.
    for i in range(n_samples):
        for j in range(n_samples):
            dots[i, j] = np.dot(x[i, :].T, x[j, :])

    # Add coefficient before returning the kernel array.
    return dots + coefficient


def _poly_kernel(x: np.ndarray, alpha: float, coefficient: float, degree: int) -> np.ndarray:
    # Check array's dimensions.
    _array_dim_check(x)

    # Get the number of features.
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


def _rbf_kernel(x: np.ndarray, sigma: float) -> np.ndarray:
    # Check array's dimensions.
    _array_dim_check(x)

    # Calculate the Euclidean distances for every pair of values.
    dists = pdist(x, 'sqeuclidean')

    # Convert the distances into a symmetric matrix, where xij is the distance of xi from xj.
    dists = squareform(dists)

    gamma = np.divide(1, np.multiply(2, np.square(sigma)))

    return np.exp(-gamma * dists)


class KPCA:
    def __init__(self, x: np.ndarray, kernel: Kernel = Kernel.RBF, alpha: float = None, coefficient: float = 0,
                 degree: int = 3, sigma: float = None, n_components: int = None):
        n_features = x.shape[1]

        if sigma is None:
            sigma = np.sqrt(n_features)

        if alpha is None:
            alpha = 1 / n_features

        if n_components is None:
            self.n_components = n_features
        else:
            self.n_components = n_components

        if kernel == Kernel.LINEAR:
            self._kernel = _linear_kernel(x, coefficient)
        elif kernel == Kernel.POLYNOMIAL:
            self._kernel = _poly_kernel(x, alpha, coefficient, degree)
        elif kernel == Kernel.RBF:
            self._kernel = _rbf_kernel(x, sigma)

    def fit(self):
        # Centering the symmetric NxN kernel matrix.
        N = self._kernel.shape[0]
        one_n = np.ones((N, N)) / N
        K = self._kernel - one_n.dot(self._kernel) - self._kernel.dot(one_n) + one_n.dot(self._kernel).dot(one_n)

        # Obtaining eigenvalues in descending order with corresponding eigenvectors from the symmetric matrix.
        eigenvalues, eigenvectors = np.linalg.eigh(K)

        # Obtaining the i eigenvectors that corresponds to the i highest eigenvalues.
        x_pc = np.column_stack((eigenvectors[:, -i] for i in range(1, self.n_components + 1)))

        return x_pc

    def test(self):
        print(self._kernel)
