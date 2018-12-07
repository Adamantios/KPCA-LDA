import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.decomposition import KernelPCA

from core import KPCA, Kernels
from unittest import TestCase

array = np.array([[1, 1, 3],
                  [3, 4, 6],
                  [3, 6, 6]])


class TestLinearKPCA(TestCase):
    kpca = KPCA(array, Kernels.LINEAR)
    print('Linear\n{}'.format(kpca.fit_transform()))


class TestPolyKPCA(TestCase):
    kpca = KPCA(array, Kernels.POLYNOMIAL)
    print('Poly\n{}'.format(kpca.fit_transform()))


class TestRbfKPCA(TestCase):
    kpca = KPCA(array, Kernels.RBF)
    print('Rbf\n{}'.format(kpca.fit_transform()))


class TestMinKernel(TestCase):
    kpca = KPCA(array, Kernels.MIN)
    print('Min\n{}'.format(kpca.fit_transform()))


class TestPlot(TestCase):
    X, y = make_moons(n_samples=100, random_state=123)

    plt.figure(figsize=(8, 6))

    plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', alpha=0.5)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', alpha=0.5)

    plt.title('A nonlinear 2D dataset')
    plt.ylabel('y coordinate')
    plt.xlabel('x coordinate')

    plt.show()

    kpca = KPCA(X, Kernels.RBF, sigma=0.1825, n_components=2)
    X_spca = kpca.fit_transform()

    plt.figure(figsize=(8, 6))
    plt.scatter(X_spca[y == 0, 0], X_spca[y == 0, 1], color='red', alpha=0.5)
    plt.scatter(X_spca[y == 1, 0], X_spca[y == 1, 1], color='blue', alpha=0.5)

    plt.title('First 2 principal components after RBF PCA')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()

    kpca = KPCA(X, Kernels.LINEAR, sigma=0.1825, n_components=1)
    X_spca = kpca.fit_transform()

    plt.figure(figsize=(8, 6))
    plt.scatter(X_spca[y == 0, 0], np.zeros((50, 1)), color='red', alpha=0.5)
    plt.scatter(X_spca[y == 1, 0], np.zeros((50, 1)), color='blue', alpha=0.5)

    plt.title('First principal component after RBF PCA')
    plt.xlabel('PC1')

    plt.show()
