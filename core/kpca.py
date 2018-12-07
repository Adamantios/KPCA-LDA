from enum import Enum, auto


def _linear_kernel():
    return 1


def _poly_kernel():
    return 2


def _rbf_kernel():
    return 3


class Kernel(Enum):
    LINEAR = _linear_kernel()
    POLYNOMIAL = _poly_kernel()
    RBF = _rbf_kernel()


class KPCA:
    def __init__(self, kernel: Kernel):
        self._kernel = kernel.value

    def test(self):
        print(self._kernel)
