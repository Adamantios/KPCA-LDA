import numpy as np
from enum import Enum, auto
from typing import NoReturn, Union

from scipy.spatial.distance import pdist, squareform

KernelReturnType = Union[np.ndarray, NoReturn]


class Kernel(Enum):
    LINEAR = auto()
    POLYNOMIAL = auto()
    RBF = auto()


def _array_dim_check(x: np.ndarray):
    if x.ndim != 2:
        raise AttributeError('Input array should be 2 dimensional.')


def _linear_kernel(x: np.ndarray, coefficient: float) -> KernelReturnType:
    # Check array's dimensions.
    _array_dim_check(x)

    # Calculate the Euclidean distances for every pair of values.
    dists = pdist(x, 'sqeuclidean')

    # Convert the distances into a symmetric matrix, where xij is the distance of xi from xj.
    dists = squareform(dists)

    # Add coefficient before returning the kernel array.
    return dists + coefficient


def _poly_kernel(x: np.ndarray, alpha: float, coefficient: float, degree: int):
    # Check array's dimensions.
    _array_dim_check(x)

    # Calculate the Euclidean distances for every pair of values.
    dists = pdist(x, 'sqeuclidean')

    # Convert the distances into a symmetric matrix, where xij is the distance of xi from xj.
    dists = squareform(dists)

    return np.power(alpha * dists + coefficient, degree)


def _rbf_kernel(x: np.ndarray, sigma: float):
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
                 degree: int = 3, sigma: float = None):
        n_samples, n_features = x.shape

        if sigma is None:
            sigma = np.sqrt(n_features)

        if alpha is None:
            alpha = 1 / n_features

        if kernel == Kernel.LINEAR:
            self._kernel = _linear_kernel(x, coefficient)
        elif kernel == Kernel.POLYNOMIAL:
            self._kernel = _poly_kernel(x, alpha, coefficient, degree)
        elif kernel == Kernel.RBF:
            self._kernel = _rbf_kernel(x, sigma)

    def test(self):
        print(self._kernel)
