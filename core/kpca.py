import numpy as np
from enum import Enum, auto
from typing import NoReturn, Union

from scipy.spatial.distance import pdist

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

    # Add coefficient before returning the kernel array.
    return dists + coefficient


def _poly_kernel(x: np.ndarray):
    _array_dim_check(x)

    return x


def _rbf_kernel(x: np.ndarray):
    _array_dim_check(x)

    return x


class KPCA:
    def __init__(self, kernel: Kernel, x: np.ndarray, coefficient: float):
        if kernel == Kernel.LINEAR:
            self._kernel = _linear_kernel(x, coefficient)

    def test(self):
        print(self._kernel)
