import numpy as np
from sklearn.datasets import make_moons
from sklearn.decomposition import KernelPCA

from core import KPCA, Kernels
from unittest import TestCase

from helpers import Plotter

array = np.array([[16, 12],
                  [93, 43],
                  [13, 46],
                  [3, 8]])

plotter = Plotter()


class TestLinearKPCA(TestCase):
    kpca = KPCA(Kernels.LINEAR)
    my_results = kpca.fit_transform(array)
    print('Linear \n{}\n{}\n'.format(kpca.alphas, kpca.lambdas))
    scikit_pca = KernelPCA(coef0=0)
    scikit_results = scikit_pca.fit_transform(array)
    print('Scikit Linear \n{}\n{}\n'.format(scikit_pca.alphas_, scikit_pca.lambdas_))

    print('Mine \n{}'.format(my_results))
    print('Scikit\'s \n{}'.format(scikit_results))

    plotter.scatter_pcs(kpca.alphas[:, 0], np.array([0, 1, 1, 0]))


class TestPolyKPCA(TestCase):
    kpca = KPCA(Kernels.POLYNOMIAL)
    my_results = kpca.fit_transform(array)
    print('Poly \n{}\n{}\n'.format(kpca.alphas, kpca.lambdas))
    scikit_pca = KernelPCA(kernel='poly', coef0=0, n_components=4)
    scikit_results = scikit_pca.fit_transform(array)
    print('Scikit Poly \n{}\n{}\n'.format(scikit_pca.alphas_, scikit_pca.lambdas_))

    print('Mine \n{}'.format(my_results))
    print('Scikit\'s \n{}'.format(scikit_results))


class TestRbfKPCA(TestCase):
    kpca = KPCA(Kernels.RBF)
    my_results = kpca.fit_transform(array)
    print('Rbf \n{}\n{}\n'.format(kpca.alphas, kpca.lambdas))
    scikit_pca = KernelPCA(kernel='rbf')
    scikit_results = scikit_pca.fit_transform(array)
    print('Scikit Rbf \n{}\n{}\n'.format(scikit_pca.alphas_, scikit_pca.lambdas_))

    print('Mine \n{}'.format(my_results))
    print('Scikit\'s \n{}'.format(scikit_results))


class TestMinKernel(TestCase):
    kpca = KPCA(Kernels.MIN)
    my_results = kpca.fit_transform(array)
    print('Min \n{}\n{}\n'.format(kpca.alphas, kpca.lambdas))

    print('Mine \n{}'.format(my_results))


class TestPlot(TestCase):
    x, y = make_moons(n_samples=100, random_state=123)

    plotter.scatter_pcs(x, y)

    kpca = KPCA(Kernels.RBF, sigma=0.17, n_components=2)
    kpca.fit_transform(x)

    plotter.scatter_pcs(kpca.alphas, y)

    kpca = KPCA(Kernels.RBF, sigma=0.17, n_components=1)
    kpca.fit_transform(x)

    plotter.scatter_pcs(kpca.alphas, y)

    from matplotlib import pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.scatter(kpca.alphas[y == 0, 0], np.zeros(50), color='red', alpha=0.5)
    plt.scatter(kpca.alphas[y == 1, 0], np.zeros(50), color='blue', alpha=0.5)
    plt.scatter(kpca.alphas[25], 0, color='black', label='original projection of point X[24]', marker='^', s=100)
    plt.scatter(kpca.transform(x[25]), 0, color='green', label='remapped point X[24]', marker='x', s=500)
    plt.legend(scatterpoints=1)
    plt.show()
