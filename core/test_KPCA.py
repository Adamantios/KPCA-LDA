import numpy as np
from core import KPCA, Kernel
from unittest import TestCase

array = np.array([[1, 1, 3],
                  [3, 4, 6],
                  [3, 6, 6]])


class TestLinearKPCA(TestCase):
    kpca = KPCA(array, Kernel.LINEAR)
    print('Linear\n{}'.format(kpca.fit()))


class TestPolyKPCA(TestCase):
    kpca = KPCA(array, Kernel.POLYNOMIAL)
    print('Poly\n{}'.format(kpca.fit()))


class TestRbfKPCA(TestCase):
    kpca = KPCA(array, Kernel.RBF)
    print('Rbf\n{}'.format(kpca.fit()))
