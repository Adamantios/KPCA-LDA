import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.decomposition import KernelPCA

from core import KPCA, Kernels
from unittest import TestCase

from helpers import Plotter

array = np.array([[16, 12, 36],
                  [93, 43, 64],
                  [13, 46, 86],
                  [3, 8, 6]])

plotter = Plotter()


class TestLinearKPCA(TestCase):
    kpca = KPCA(Kernels.LINEAR)
    print('Linear\n{}'.format(kpca.fit_transform(array)))
    scikit_pca = KernelPCA()
    scikit_pca.fit_transform(array)
    print('Scikit Linear {}\n{}\n'.format(scikit_pca.alphas_, scikit_pca.lambdas_))


class TestPolyKPCA(TestCase):
    kpca = KPCA(Kernels.POLYNOMIAL)
    print('Poly\n{}'.format(kpca.fit_transform(array)))
    scikit_pca = KernelPCA(kernel='poly')
    scikit_pca.fit_transform(array)
    print('Scikit Poly {}\n{}\n'.format(scikit_pca.alphas_, scikit_pca.lambdas_))


class TestRbfKPCA(TestCase):
    kpca = KPCA(Kernels.RBF)
    print('Rbf\n{}'.format(kpca.fit_transform(array)))
    scikit_pca = KernelPCA(kernel='rbf')
    scikit_pca.fit_transform(array)
    print('Scikit Rbf {}\n{}\n'.format(scikit_pca.alphas_, scikit_pca.lambdas_))


class TestMinKernel(TestCase):
    kpca = KPCA(Kernels.MIN)
    print('Min\n{}'.format(kpca.fit_transform(array)))


class TestPlot(TestCase):
    x, y = make_moons(n_samples=100, random_state=123)

    plotter.scatter_pcs(x, y)

    kpca = KPCA(Kernels.RBF, sigma=0.17, n_components=2)
    x_spca = kpca.fit_transform(x)[0]

    plotter.scatter_pcs(x_spca, y)

    kpca = KPCA(Kernels.RBF, sigma=0.17, n_components=1)
    x_spca = kpca.fit_transform(x)[0]

    plotter.scatter_pcs(x_spca, y)
